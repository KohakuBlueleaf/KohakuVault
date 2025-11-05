// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KVault implementation - Key-value storage with caching
//!
//! This module provides a key-value store backed by SQLite with:
//! - Write-back caching for improved write performance
//! - Streaming support for large values via BLOB API
//! - Flexible key types (bytes or strings)

mod ops;
mod stream;

use crate::common::{checkpoint_wal, open_connection, VaultError, WriteBackCache};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, Connection};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Main KVault struct exposed to Python.
/// Provides a dict-like interface for key-value storage.
#[pyclass]
pub struct _KVault {
    pub(crate) conn: Mutex<Connection>,
    pub(crate) table: String,
    pub(crate) cache: Mutex<Option<WriteBackCache<Vec<u8>, Vec<u8>>>>,
    pub(crate) chunk_size: usize,
    pub(crate) cache_locked: Arc<AtomicBool>, // For lock_cache()
}

// Python-exposed methods (single #[pymethods] block required by PyO3)
#[pymethods]
impl _KVault {
    /// Create a new KVault instance.
    ///
    /// # Arguments
    /// * `path` - SQLite file path
    /// * `table` - Table name (default "kv")
    /// * `chunk_size` - Streaming chunk size in bytes (default 1 MiB)
    /// * `enable_wal` - Enable WAL mode for concurrent access (default true)
    /// * `page_size` - SQLite page size (default 32KB)
    /// * `mmap_size` - Memory-mapped I/O size (default 256MB)
    /// * `cache_kb` - SQLite cache size in KB (default 100MB)
    #[new]
    #[pyo3(signature = (path, table="kv", chunk_size=1<<20, enable_wal=true, page_size=32768, mmap_size=268_435_456, cache_kb=100_000))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python<'_>,
        path: &str,
        table: &str,
        chunk_size: usize,
        enable_wal: bool,
        page_size: u32,
        mmap_size: u64,
        cache_kb: i64,
    ) -> PyResult<Self> {
        let conn = open_connection(path, enable_wal, page_size, mmap_size, cache_kb)
            .map_err(VaultError::from)?;

        // Create schema
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS {t} (
                id    INTEGER PRIMARY KEY,
                key   BLOB    NOT NULL UNIQUE,
                value BLOB    NOT NULL,
                size  INTEGER NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS {t}_key_idx
            ON {t}(key);
            ",
            t = rusqlite::types::ValueRef::from(table)
                .as_str()
                .unwrap_or("kv") // defensive
        ))
        .map_err(VaultError::from)?;

        Ok(Self {
            conn: Mutex::new(conn),
            table: table.to_string(),
            cache: Mutex::new(None),
            chunk_size,
            cache_locked: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Enable write-back cache (bytes-bounded). Flush when threshold reached.
    /// Daemon thread for auto-flush is handled in Python layer.
    ///
    /// # Arguments
    /// * `cap_bytes` - Maximum cache size in bytes (default 64MB)
    /// * `flush_threshold` - Auto-flush trigger point (default 16MB)
    /// * `_flush_interval` - Ignored (handled by Python daemon thread)
    #[pyo3(signature = (cap_bytes=64<<20, flush_threshold=16<<20, _flush_interval=None))]
    fn enable_cache(&self, cap_bytes: usize, flush_threshold: usize, _flush_interval: Option<f64>) {
        let mut guard = self.cache.lock().unwrap();
        *guard = Some(WriteBackCache::new(cap_bytes, flush_threshold));
        // Note: flush_interval is used by Python daemon thread, not Rust
    }

    /// Disable cache (auto-flushes first).
    fn disable_cache(&self, _py: Python<'_>) -> PyResult<()> {
        // Flush before disabling
        self.flush_cache(_py)?;

        let mut guard = self.cache.lock().unwrap();
        *guard = None;
        Ok(())
    }

    /// Flush write-back cache (if enabled) in a single transaction.
    /// Respects cache_locked flag - won't flush if locked.
    ///
    /// # Returns
    /// Number of entries flushed
    fn flush_cache(&self, _py: Python<'_>) -> PyResult<usize> {
        // Check if cache is locked (for lock_cache() context manager)
        if self.cache_locked.load(Ordering::Relaxed) {
            return Ok(0); // Skip flush if locked
        }

        let mut guard = self.cache.lock().unwrap();
        let Some(cache) = guard.as_mut() else {
            return Ok(0);
        };

        // Don't flush if empty
        if cache.is_empty() {
            return Ok(0);
        }

        let entries = cache.drain();
        drop(guard); // Release lock before transaction

        let sql = format!(
            "
            INSERT INTO {t}(key, value, size)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                size = excluded.size
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(VaultError::from)?;
        let mut stmt = tx.prepare(&sql).map_err(VaultError::from)?;
        let mut count = 0usize;
        for (k, v) in entries {
            stmt.execute(params![k, &v, (v.len() as i64)])
                .map_err(VaultError::from)?;
            count += 1;
        }
        drop(stmt);
        tx.commit().map_err(VaultError::from)?;
        Ok(count)
    }

    /// Vacuum & optimize (blocks writer).
    fn optimize(&self, _py: Python<'_>) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("PRAGMA optimize; VACUUM;")
            .map_err(VaultError::from)?;
        Ok(())
    }

    /// Get number of keys in the store.
    fn len(&self, _py: Python<'_>) -> PyResult<i64> {
        let sql = format!(
            "
            SELECT COUNT(*)
            FROM {}
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let n: i64 = conn
            .query_row(&sql, [], |r| r.get(0))
            .map_err(VaultError::from)?;
        Ok(n)
    }

    /// Set cache lock status (for Python lock_cache() context manager).
    fn set_cache_locked(&self, locked: bool) {
        self.cache_locked.store(locked, Ordering::Relaxed);
    }

    /// Manually checkpoint WAL file to main database.
    /// This helps prevent WAL from growing too large.
    ///
    /// # Returns
    /// Success indicator (0 on success)
    fn checkpoint_wal(&self) -> PyResult<i64> {
        let conn = self.conn.lock().unwrap();
        checkpoint_wal(&conn).map_err(|e| e.into())
    }

    // ===== Core Operations =====

    fn put(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.put_impl(py, key, value)
    }

    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        self.get_impl(py, key)
    }

    fn delete(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.delete_impl(py, key)
    }

    fn exists(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.exists_impl(py, key)
    }

    #[pyo3(signature = (prefix=None, limit=1000))]
    fn scan_keys(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: usize,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        self.scan_keys_impl(py, prefix, limit)
    }

    // ===== Streaming Operations =====

    #[pyo3(signature = (key, reader, size, chunk_size=None))]
    fn put_stream(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        reader: &Bound<'_, PyAny>,
        size: usize,
        chunk_size: Option<usize>,
    ) -> PyResult<()> {
        self.put_stream_impl(py, key, reader, size, chunk_size)
    }

    #[pyo3(signature = (key, writer, chunk_size=None))]
    fn get_to_file(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        writer: &Bound<'_, PyAny>,
        chunk_size: Option<usize>,
    ) -> PyResult<usize> {
        self.get_to_file_impl(py, key, writer, chunk_size)
    }
}

// Helper methods (not exposed to Python)
impl _KVault {
    /// Write directly to database (bypass cache).
    pub(crate) fn write_direct(&self, k: &[u8], v: &[u8]) -> PyResult<()> {
        let size = v.len() as i64;
        let sql = format!(
            "
            INSERT INTO {t}(key, value, size)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                size = excluded.size
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        conn.execute(&sql, params![k, v, size])
            .map_err(VaultError::from)?;
        Ok(())
    }
}
