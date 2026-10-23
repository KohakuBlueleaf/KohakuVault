//! Atomic mapping operations; Python conversion never runs under database locks.

use pyo3::prelude::*;
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use std::sync::{atomic::Ordering, Arc};

use super::{_KVault, cache::SharedValue, ops::to_key_bytes};
use crate::common::VaultError;

impl _KVault {
    fn pending_value(&self, key: &[u8]) -> Option<SharedValue> {
        self.cache
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|cache| cache.get(key))
    }

    pub(crate) fn clear_impl(&self, py: Python<'_>) -> PyResult<()> {
        self.ensure_writable()?;
        Ok(py.allow_threads(|| -> Result<(), VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let mut conn = self.connection()?;
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute(&format!("DELETE FROM {}", self.table), [])?;
            tx.commit()?;
            let retired = self
                .cache
                .lock()
                .unwrap()
                .as_mut()
                .map(|cache| cache.take_pending());
            drop(retired);
            Ok(())
        })?)
    }

    pub(crate) fn pop_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.ensure_writable()?;
        let key = to_key_bytes(py, key)?;
        let value = py.allow_threads(|| -> Result<Option<SharedValue>, VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let mut conn = self.connection()?;
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let value = match self.pending_value(&key) {
                Some(value) => Some(value),
                None => tx
                    .prepare_cached(&format!("SELECT value FROM {} WHERE key=?1", self.table))?
                    .query_row([&key], |row| row.get::<_, Vec<u8>>(0))
                    .optional()?
                    .map(Arc::new),
            };
            if value.is_some() {
                tx.execute(&format!("DELETE FROM {} WHERE key=?1", self.table), [&key])?;
            }
            tx.commit()?;
            self.remove_pending(&key);
            Ok(value)
        })?;
        match value {
            Some(value) => self.decode_result(py, &value),
            None => Err(pyo3::exceptions::PyKeyError::new_err("Key not found")),
        }
    }

    pub(crate) fn setdefault_impl(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: PyObject,
    ) -> PyResult<PyObject> {
        self.ensure_open()?;
        let key = to_key_bytes(py, key)?;
        // Avoid converting an unused default (which may not be serializable).
        let existing = py.allow_threads(|| -> Result<Option<SharedValue>, VaultError> {
            if let Some(value) = self.pending_value(&key) {
                return Ok(Some(value));
            }
            self.with_point_reader(|conn| {
                Ok(conn
                    .prepare_cached(&format!("SELECT value FROM {} WHERE key=?1", self.table))?
                    .query_row([&key], |row| row.get::<_, Vec<u8>>(0))
                    .optional()?
                    .map(Arc::new))
            })
        })?;
        if let Some(value) = existing {
            return self.decode_result(py, &value);
        }
        self.ensure_writable()?;
        let encoded = Arc::new(self.pack_value(py, default.bind(py))?);
        let value = py.allow_threads(|| -> Result<SharedValue, VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let mut conn = self.connection()?;
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            // Recheck after serialization and writer reservation for independent callers.
            let existing = match self.pending_value(&key) {
                Some(value) => Some(value),
                None => tx
                    .prepare_cached(&format!("SELECT value FROM {} WHERE key=?1", self.table))?
                    .query_row([&key], |row| row.get::<_, Vec<u8>>(0))
                    .optional()?
                    .map(Arc::new),
            };
            if let Some(value) = existing {
                return Ok(value);
            }
            if self.cache_locked.load(Ordering::Relaxed) && self.cache.lock().unwrap().is_some() {
                return Err(VaultError::Cache(
                    "setdefault insertion prohibited while cache is locked".into(),
                ));
            }
            tx.execute(
                &format!("INSERT INTO {} (key,value) VALUES (?1,?2)", self.table),
                params![&key, encoded.as_slice()],
            )?;
            tx.commit()?;
            Ok(encoded)
        })?;
        self.decode_result(py, &value)
    }
}
