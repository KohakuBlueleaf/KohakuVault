// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KVault implementation - Key-value storage with caching
//!
//! This module provides a key-value store backed by SQLite with:
//! - Write-back caching for improved write performance
//! - Streaming support for large values via BLOB API
//! - Flexible key types (bytes or strings)
//! - Optional header system for encoding type detection
//! - Auto-packing for arbitrary Python objects

pub(crate) mod autopacker;
mod encoding;
pub(crate) mod header;
mod ops;
mod stream;

pub(crate) use autopacker::AutoPacker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, Connection};

use crate::common::{checkpoint_wal, meta::MetaTable, open_connection, VaultError, WriteBackCache};

/// Main KVault struct exposed to Python.
/// Provides a dict-like interface for key-value storage.
#[pyclass]
pub struct _KVault {
    pub(crate) conn: Mutex<Connection>,
    pub(crate) table: String,
    pub(crate) cache: Mutex<Option<WriteBackCache<Vec<u8>, Vec<u8>>>>,
    pub(crate) chunk_size: usize,
    pub(crate) cache_locked: Arc<AtomicBool>, // For lock_cache()
    pub(crate) use_headers: AtomicBool,       // Enable header format for new writes
    pub(crate) auto_packer: Mutex<Option<AutoPacker>>, // Auto-packing for arbitrary objects
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

        // Create metadata table (shared with ColumnVault)
        MetaTable::ensure_table(&conn).map_err(VaultError::from)?;

        // Create schema - minimal design with key as primary key
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS {t} (
                key   BLOB PRIMARY KEY NOT NULL,
                value BLOB NOT NULL
            );
            ",
            t = rusqlite::types::ValueRef::from(table)
                .as_str()
                .unwrap_or("kv") // defensive
        ))
        .map_err(VaultError::from)?;

        // Check if header feature is registered (indicates this DB has been used with headers)
        let _headers_supported = MetaTable::has_feature(
            &conn,
            crate::common::meta::KV_FEATURES_KEY,
            crate::common::meta::KV_FEATURE_HEADERS,
        )
        .unwrap_or(false);
        // Note: Always start with headers disabled for backward compat
        // User must explicitly enable if needed

        Ok(Self {
            conn: Mutex::new(conn),
            table: table.to_string(),
            cache: Mutex::new(None),
            chunk_size,
            cache_locked: Arc::new(AtomicBool::new(false)),
            use_headers: AtomicBool::new(true), // DEFAULT: headers enabled for auto-packing
            auto_packer: Mutex::new(Some(AutoPacker::new(true))), // DEFAULT: auto-pack enabled
        })
    }

    /// Enable header format for new writes
    ///
    /// When enabled, all new values will be written with a 10-byte header
    /// that indicates the encoding type. This allows for:
    /// - Auto-packing of Python objects (future)
    /// - Mixed encoding types in same vault
    /// - Compression/encryption flags (future)
    ///
    /// Note: Existing values without headers are still readable (backward compatible)
    fn enable_headers(&self) -> PyResult<()> {
        self.use_headers.store(true, Ordering::Relaxed);

        // Register feature in meta table
        let conn = self.conn.lock().unwrap();
        MetaTable::register_feature(
            &conn,
            crate::common::meta::KV_FEATURES_KEY,
            crate::common::meta::KV_FEATURE_HEADERS,
        )
        .map_err(VaultError::from)?;

        Ok(())
    }

    /// Disable header format (return to raw bytes mode)
    fn disable_headers(&self) {
        self.use_headers.store(false, Ordering::Relaxed);
    }

    /// Check if headers are enabled
    fn headers_enabled(&self) -> bool {
        self.use_headers.load(Ordering::Relaxed)
    }

    /// Enable auto-packing (allows arbitrary Python objects)
    ///
    /// When enabled, KVault automatically serializes Python objects:
    /// - numpy arrays → DataPacker vec:*
    /// - dicts/lists → MessagePack
    /// - int/float → DataPacker i64/f64
    /// - bytes → Raw (no header)
    /// - Custom objects → Pickle (if use_pickle=True)
    #[pyo3(signature = (use_pickle=true))]
    fn enable_auto_pack(&self, use_pickle: bool) -> PyResult<()> {
        let mut guard = self.auto_packer.lock().unwrap();
        *guard = Some(AutoPacker::new(use_pickle));

        // Also enable headers (required for auto-pack)
        self.use_headers.store(true, Ordering::Relaxed);

        // Register features in meta table
        let conn = self.conn.lock().unwrap();
        MetaTable::register_feature(
            &conn,
            crate::common::meta::KV_FEATURES_KEY,
            crate::common::meta::KV_FEATURE_AUTO_PACK,
        )
        .map_err(VaultError::from)?;

        Ok(())
    }

    /// Disable auto-packing (return to bytes-only mode)
    fn disable_auto_pack(&self) {
        let mut guard = self.auto_packer.lock().unwrap();
        *guard = None;
    }

    /// Check if auto-packing is enabled
    fn auto_pack_enabled(&self) -> bool {
        self.auto_packer.lock().unwrap().is_some()
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
            INSERT INTO {t}(key, value)
            VALUES (?1, ?2)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(VaultError::from)?;
        let mut stmt = tx.prepare(&sql).map_err(VaultError::from)?;
        let mut count = 0usize;
        for (k, v) in entries {
            stmt.execute(params![k, &v]).map_err(VaultError::from)?;
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

    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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
        let sql = format!(
            "
            INSERT INTO {t}(key, value)
            VALUES (?1, ?2)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        conn.execute(&sql, params![k, v])
            .map_err(VaultError::from)?;
        Ok(())
    }
}
