// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Core KVault operations implementation functions
//!
//! This module contains the actual implementation of KVault operations.
//! The main KVault struct in mod.rs provides thin wrappers that delegate to these functions.

use crate::common::{CacheError, VaultError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyString, PyStringMethods};
use rusqlite::params;

use super::_KVault;

/// Convert Python key (bytes or str) to bytes.
pub(crate) fn to_key_bytes(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<Vec<u8>, VaultError> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        Ok(b.as_bytes().to_vec())
    } else if let Ok(s) = obj.downcast::<PyString>() {
        Ok(s.to_str()
            .map_err(|e| VaultError::Py(format!("Invalid UTF-8: {}", e)))?
            .as_bytes()
            .to_vec())
    } else {
        Err(VaultError::Py("key must be bytes or str".into()))
    }
}

impl _KVault {
    /// Insert/replace whole value (bytes-like). Uses UPSERT and sets size.
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    /// * `value` - Value (bytes)
    ///
    /// # Behavior
    /// - If cache is enabled: tries to cache the value
    /// - If value too large or cache full: flushes and writes directly
    /// - Auto-flushes when threshold is reached
    pub(crate) fn put_impl(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let k = to_key_bytes(py, key)?;
        let v = if let Ok(b) = value.downcast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else {
            return Err(VaultError::Py("value must be bytes-like".into()).into());
        };

        // Try to use cache if enabled
        {
            let mut guard = self.cache.lock().unwrap();
            if let Some(cache) = guard.as_mut() {
                match cache.insert(k.clone(), v.clone()) {
                    Ok(()) => {
                        // Successfully cached, check if should auto-flush
                        let should_flush = cache.should_flush();
                        drop(guard); // Release lock before flush

                        if should_flush {
                            self.flush_cache(py)?;
                        }
                        return Ok(());
                    }
                    Err(CacheError::ValueTooLarge) => {
                        // Value too large for cache, flush existing then bypass cache
                        drop(guard);
                        self.flush_cache(py)?;
                        // Fall through to direct write
                    }
                    Err(CacheError::NeedFlush) => {
                        // Cache full, flush then retry insert
                        drop(guard);
                        self.flush_cache(py)?;

                        // Retry insert after flush
                        let mut guard = self.cache.lock().unwrap();
                        if let Some(cache) = guard.as_mut() {
                            cache.insert(k, v).ok(); // Should succeed now
                        }
                        return Ok(());
                    }
                }
            }
        }

        // Direct write (no cache or bypassed for large value)
        self.write_direct(&k, &v)?;
        Ok(())
    }

    /// Get entire value as bytes (avoid for huge blobs; prefer get_to_file()).
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    ///
    /// # Returns
    /// Value as bytes
    ///
    /// # Errors
    /// Returns KeyError if key not found
    pub(crate) fn get_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        let k = to_key_bytes(py, key)?;

        // Check write-back cache first
        if let Some(cache) = self.cache.lock().unwrap().as_ref() {
            if let Some(v) = cache.map.get(&k) {
                return Ok(PyBytes::new_bound(py, v).unbind());
            }
        }

        let sql = format!(
            "
            SELECT value
            FROM {}
            WHERE key = ?1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let data: Vec<u8> =
            conn.query_row(&sql, params![k], |r| r.get(0))
                .map_err(|e| -> PyErr {
                    match e {
                        rusqlite::Error::QueryReturnedNoRows => {
                            VaultError::NotFound("Key not found".to_string()).into()
                        }
                        other => VaultError::from(other).into(),
                    }
                })?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Delete a key-value pair.
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    ///
    /// # Returns
    /// true if key was deleted, false if key didn't exist
    pub(crate) fn delete_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = to_key_bytes(py, key)?;
        let sql = format!(
            "
            DELETE FROM {}
            WHERE key = ?1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(&sql, params![k]).map_err(VaultError::from)?;
        Ok(n > 0)
    }

    /// Check if a key exists.
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    ///
    /// # Returns
    /// true if key exists, false otherwise
    pub(crate) fn exists_impl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = to_key_bytes(py, key)?;
        let sql = format!(
            "
            SELECT 1
            FROM {}
            WHERE key = ?1
            LIMIT 1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let found: Result<i32, _> = conn.query_row(&sql, params![k], |r| r.get(0));
        Ok(found.is_ok())
    }

    /// Scan keys (optionally prefix for TEXT-ish keys; for binary prefixes, pass bytes).
    ///
    /// # Arguments
    /// * `prefix` - Optional key prefix for filtering (bytes or str)
    /// * `limit` - Maximum number of keys to return (default 1000)
    ///
    /// # Returns
    /// List of keys (as bytes)
    pub(crate) fn scan_keys_impl(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: usize,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let mut out = Vec::new();
        let conn = self.conn.lock().unwrap();

        if let Some(p) = prefix {
            let k = to_key_bytes(py, p)?;
            // Simple prefix scan with range [prefix, prefix||0xFF...].
            // Works for raw bytes because SQLite compares blobs lexicographically.
            let mut hi = k.clone();
            hi.push(0xFF);
            let sql = format!(
                "
                SELECT key
                FROM {}
                WHERE key >= ?1 AND key < ?2
                ORDER BY key
                LIMIT ?3
                ",
                self.table
            );
            let mut stmt = conn.prepare(&sql).map_err(VaultError::from)?;
            let iter = stmt
                .query_map(params![k, hi, limit as i64], |r| r.get::<_, Vec<u8>>(0))
                .map_err(VaultError::from)?;
            for r in iter {
                let kb = r.map_err(VaultError::from)?;
                out.push(PyBytes::new_bound(py, &kb).unbind());
            }
        } else {
            let sql = format!(
                "
                SELECT key
                FROM {}
                ORDER BY key
                LIMIT ?1
                ",
                self.table
            );
            let mut stmt = conn.prepare(&sql).map_err(VaultError::from)?;
            let iter = stmt
                .query_map(params![limit as i64], |r| r.get::<_, Vec<u8>>(0))
                .map_err(VaultError::from)?;
            for r in iter {
                let kb = r.map_err(VaultError::from)?;
                out.push(PyBytes::new_bound(py, &kb).unbind());
            }
        }
        Ok(out)
    }
}
