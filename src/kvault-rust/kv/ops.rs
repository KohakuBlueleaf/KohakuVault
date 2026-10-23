//! KV operations using unchanged value encodings and independent read snapshots.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use rusqlite::{params, OptionalExtension};

use super::{_KVault, cache::SharedValue};
use crate::common::{CacheError, VaultError};

pub(crate) fn to_key_bytes(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<Vec<u8>, VaultError> {
    if let Ok(bytes) = obj.downcast::<PyBytes>() {
        Ok(bytes.as_bytes().to_vec())
    } else if let Ok(text) = obj.downcast::<PyString>() {
        Ok(text
            .to_str()
            .map_err(|error| VaultError::Py(error.to_string()))?
            .as_bytes()
            .to_vec())
    } else {
        Err(VaultError::Py("key must be bytes or str".into()))
    }
}

impl _KVault {
    pub(crate) fn put_impl(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        let key = to_key_bytes(py, key)?;
        let data = self.pack_value(py, value)?;

        let data = Arc::new(data);
        // Avoid a GIL round trip for uncontended memory-only cache admission.
        let admitted = if let Ok(mutation) = self.mutation_gate.try_read() {
            self.ensure_open()?;
            let accepted = if let Ok(mut guard) = self.cache.try_lock() {
                guard
                    .as_mut()
                    .and_then(|cache| cache.insert(&key, &data).ok().map(|_| cache.should_flush()))
            } else {
                None
            };
            drop(mutation);
            accepted
        } else {
            None
        };
        if let Some(flush) = admitted {
            if flush {
                py.allow_threads(|| self.flush_cache_native())?;
            }
            return Ok(());
        }
        py.allow_threads(|| -> PyResult<()> {
            // Retry cache admission if another writer fills it during a flush.
            loop {
                let mutation = self.mutation_gate.read().unwrap();
                self.ensure_open()?;
                let mut guard = self.cache.lock().unwrap();
                let Some(cache) = guard.as_mut() else { break };
                match cache.insert(&key, &data) {
                    Ok(()) => {
                        let flush = cache.should_flush();
                        drop(guard);
                        drop(mutation);
                        if flush {
                            self.flush_cache_native()?;
                        }
                        return Ok(());
                    }
                    Err(CacheError::ValueTooLarge) => break,
                    Err(CacheError::NeedFlush) => {
                        if self.cache_locked.load(std::sync::atomic::Ordering::Relaxed) {
                            return Err(VaultError::Cache(
                                "cache is full while flushing is locked".into(),
                            )
                            .into());
                        }
                        drop(guard);
                        drop(mutation);
                        self.flush_cache_native()?;
                    }
                }
            }
            self.write_direct(&key, &data)
        })
    }

    pub(crate) fn get_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.ensure_open()?;
        let key = to_key_bytes(py, key)?;
        let cached = self
            .cache
            .try_lock()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|cache| cache.get(&key)));
        if let Some(value) = cached {
            return self.decode_result(py, &value);
        }
        let data = py.allow_threads(|| -> Result<SharedValue, VaultError> {
            let cached = self
                .cache
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|cache| cache.get(&key));
            if let Some(value) = cached {
                return Ok(value);
            }

            self.with_point_reader(|conn| {
                let sql = format!("SELECT value FROM {} WHERE key = ?1", self.table);
                let value = conn
                    .prepare_cached(&sql)?
                    .query_row([&key], |row| row.get(0))
                    .optional()?;
                value
                    .map(Arc::new)
                    .ok_or_else(|| VaultError::NotFound("Key not found".into()))
            })
        })?;
        self.decode_result(py, &data)
    }

    pub(crate) fn delete_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.ensure_writable()?;
        let key = to_key_bytes(py, key)?;
        Ok(py.allow_threads(|| -> Result<bool, VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            // Connection ownership resolves older in-flight work before deletion.
            let conn = self.connection()?;
            let sql = format!("DELETE FROM {} WHERE key = ?1", self.table);
            let deleted = conn.prepare_cached(&sql)?.execute([&key])?;
            let pending = self
                .cache
                .lock()
                .unwrap()
                .as_mut()
                .and_then(|cache| cache.remove(&key));
            Ok(deleted > 0 || pending.is_some())
        })?)
    }

    pub(crate) fn exists_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.ensure_open()?;
        let key = to_key_bytes(py, key)?;
        Ok(py.allow_threads(|| -> Result<bool, VaultError> {
            let pending = self
                .cache
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|cache| cache.contains_key(&key));
            if pending {
                return Ok(true);
            }
            self.with_point_reader(|conn| {
                let sql = format!("SELECT 1 FROM {} WHERE key = ?1", self.table);
                Ok(conn
                    .prepare_cached(&sql)?
                    .query_row(params![key], |_| Ok(()))
                    .optional()?
                    .is_some())
            })
        })?)
    }
}
