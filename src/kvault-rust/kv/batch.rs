//! Explicit, eagerly materialized batches. No Python callbacks run under SQL locks.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use super::_KVault;
use super::ops::to_key_bytes;
use crate::common::VaultError;

/// Copy references using the builtin container APIs, never arbitrary iteration.
/// Keeping owned references also insulates serialization from list mutations in
/// a user's pickle callback.
fn explicit_batch(value: &Bound<'_, PyAny>, description: &str) -> PyResult<Vec<PyObject>> {
    if let Ok(list) = value.downcast::<PyList>() {
        Ok(list.iter().map(Bound::unbind).collect())
    } else if let Ok(tuple) = value.downcast::<PyTuple>() {
        Ok(tuple.iter().map(Bound::unbind).collect())
    } else {
        Err(PyTypeError::new_err(format!("{description} must be a list or tuple")))
    }
}

impl _KVault {
    pub(crate) fn put_many_impl(&self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        // Even an empty batch must reject readonly or closed vaults.
        self.ensure_writable()?;
        let items = explicit_batch(items, "items")?;
        let mut pairs = Vec::with_capacity(items.len());
        for item in &items {
            let pair = explicit_batch(item.bind(py), "each batch item")?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "each batch item must contain exactly (key, value)",
                ));
            }
            pairs.push(pair);
        }

        // Preserve the single-put pipeline, including raw bytes and headers.
        // Serialize every occurrence, including superseded duplicates: an error
        // anywhere in the explicit batch must precede any database mutation.
        let mut encoded = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let key = to_key_bytes(py, pair[0].bind(py))?;
            let value = pair[1].bind(py);
            let data = self.pack_value(py, value)?;
            encoded.push((key, data));
        }

        Ok(py.allow_threads(|| -> Result<(), VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            if self.cache_locked.load(Ordering::Relaxed) && self.cache.lock().unwrap().is_some() {
                return Err(VaultError::Cache(
                    "batch write prohibited while cache is locked".into(),
                ));
            }
            if encoded.is_empty() {
                return Ok(());
            }
            let mut conn = self.connection()?;
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let sql = format!(
                "INSERT INTO {}(key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                self.table
            );
            {
                let mut stmt = tx.prepare_cached(&sql)?;
                for (key, value) in &encoded {
                    stmt.execute(params![key, value])?;
                }
            }
            tx.commit()?;
            // Keep both mutation admission and the writer connection blocked
            // until stale pending values are gone. On any SQL/commit failure,
            // the transaction rolls back and pending entries remain untouched.
            for (key, _) in &encoded {
                self.remove_pending(key);
            }
            Ok(())
        })?)
    }

    pub(crate) fn get_many_impl(
        &self,
        py: Python<'_>,
        keys: &Bound<'_, PyAny>,
        default: Option<PyObject>,
    ) -> PyResult<Vec<PyObject>> {
        self.ensure_open()?;
        let keys = explicit_batch(keys, "keys")?;
        let keys = keys
            .iter()
            .map(|key| to_key_bytes(py, key.bind(py)))
            .collect::<Result<Vec<_>, _>>()?;

        let values = py.allow_threads(|| -> Result<Vec<Option<Arc<Vec<u8>>>>, VaultError> {
            self.ensure_open()?;
            // Clone only matching Arc handles, not the whole cache or payloads.
            let mut values: Vec<_> = {
                let guard = self.cache.lock().unwrap();
                keys.iter()
                    .map(|key| guard.as_ref().and_then(|cache| cache.get(key)))
                    .collect()
            };
            if values.iter().any(Option::is_none) {
                self.with_reader(|conn| {
                    // All committed lookups share one snapshot, not one
                    // transaction per key. Pending values take precedence.
                    let sql = format!("SELECT value FROM {} WHERE key = ?1", self.table);
                    {
                        let mut stmt = conn.prepare_cached(&sql)?;
                        for (key, value) in keys.iter().zip(values.iter_mut()) {
                            if value.is_none() {
                                let stored: Option<Vec<u8>> =
                                    stmt.query_row([key], |row| row.get(0)).optional()?;
                                *value = stored.map(Arc::new);
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(values)
        })?;

        // Deserialization can invoke Python; release all cache/SQL locks and
        // end the snapshot before doing so. Missing is distinct from stored None.
        let default = default.unwrap_or_else(|| py.None());
        values
            .into_iter()
            .map(|value| match value {
                Some(value) => self.decode_result(py, value.as_slice()),
                None => Ok(default.clone_ref(py)),
            })
            .collect()
    }
}
