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
mod batch;
mod cache;
mod encoding;
pub(crate) mod header;
mod mapping;
mod ops;
mod scan;
mod stream;

pub(crate) use autopacker::AutoPacker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, Connection};

use crate::common::read_pool::ReadPool;
use crate::common::{
    checkpoint_wal, meta::MetaTable, open_connection, open_readonly_connection, reader_pool,
    VaultError,
};

use self::cache::KvCache;

/// Main KVault struct exposed to Python.
/// Provides a dict-like interface for key-value storage.
#[pyclass]
pub struct _KVault {
    pub(crate) conn: parking_lot::Mutex<Option<Connection>>,
    pub(crate) mutation_gate: RwLock<()>,
    pub(crate) closed: AtomicBool,
    pub(crate) readers: Option<ReadPool>,
    pub(crate) read_only: bool,
    pub(crate) table: String,
    pub(crate) cache: Mutex<Option<KvCache>>,
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
    #[pyo3(signature = (path, table="kv", chunk_size=1<<20, enable_wal=true, page_size=32768, mmap_size=268_435_456, cache_kb=100_000, read_only=false, reader_cache_kb=8192, reader_idle_limit=8))]
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
        read_only: bool,
        reader_cache_kb: i64,
        reader_idle_limit: usize,
    ) -> PyResult<Self> {
        if reader_cache_kb <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "reader_cache_kb must be positive",
            ));
        }
        _py.allow_threads(|| {
            let conn = if read_only {
                open_readonly_connection(path, mmap_size, cache_kb)?
            } else {
                open_connection(path, enable_wal, page_size, mmap_size, cache_kb)?
            };

            // Reader opens must not create tables or register storage features.
            if !read_only {
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
            } else {
                conn.prepare(&format!("SELECT key, value FROM {} LIMIT 0", table))
                    .map_err(VaultError::from)?;
            }

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
                conn: parking_lot::Mutex::new(Some(conn)),
                mutation_gate: RwLock::new(()),
                closed: AtomicBool::new(false),
                readers: reader_pool(path)
                    .map(|pool| pool.configured(reader_cache_kb, mmap_size, reader_idle_limit)),
                read_only,
                table: table.to_string(),
                cache: Mutex::new(None),
                chunk_size,
                cache_locked: Arc::new(AtomicBool::new(false)),
                use_headers: AtomicBool::new(true), // DEFAULT: headers enabled for auto-packing
                auto_packer: Mutex::new(Some(AutoPacker::new(true))), // DEFAULT: auto-pack enabled
            })
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
    fn enable_headers(&self, py: Python<'_>) -> PyResult<()> {
        self.check_stream_reentry()?;
        py.allow_threads(|| {
            self.use_headers.store(true, Ordering::Relaxed);
            if self.read_only {
                return Ok(());
            }

            // Register feature in meta table
            let conn = self.connection()?;
            MetaTable::register_feature(
                &conn,
                crate::common::meta::KV_FEATURES_KEY,
                crate::common::meta::KV_FEATURE_HEADERS,
            )
            .map_err(VaultError::from)?;

            Ok(())
        })
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
    fn enable_auto_pack(&self, py: Python<'_>, use_pickle: bool) -> PyResult<()> {
        self.check_stream_reentry()?;
        let mut guard = self.auto_packer.lock().unwrap();
        *guard = Some(AutoPacker::new(use_pickle));

        // Also enable headers (required for auto-pack)
        self.use_headers.store(true, Ordering::Relaxed);

        drop(guard);
        py.allow_threads(|| {
            if self.read_only {
                return Ok(());
            }
            // Register features in meta table
            let conn = self.connection()?;
            MetaTable::register_feature(
                &conn,
                crate::common::meta::KV_FEATURES_KEY,
                crate::common::meta::KV_FEATURE_AUTO_PACK,
            )
            .map_err(VaultError::from)?;

            Ok(())
        })
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
    fn enable_cache(
        &self,
        py: Python<'_>,
        cap_bytes: usize,
        flush_threshold: usize,
        _flush_interval: Option<f64>,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        if cap_bytes == 0 || flush_threshold == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("cache sizes must be positive"));
        }
        Ok(py.allow_threads(|| -> Result<(), VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let conn = self.connection()?;
            {
                let guard = self.cache.lock().unwrap();
                if self.cache_locked.load(Ordering::Relaxed)
                    && guard.as_ref().is_some_and(|cache| !cache.is_empty())
                {
                    return Err(VaultError::Cache("cannot reset a locked nonempty cache".into()));
                }
            }
            self.flush_cache_locked(&conn)?;
            *self.cache.lock().unwrap() =
                Some(KvCache::new(cap_bytes, flush_threshold.min(cap_bytes)));
            Ok(())
        })?)
    }

    /// Stop mutation admission only after a successful final flush.
    fn finish_close(&self, py: Python<'_>) -> PyResult<()> {
        self.check_stream_reentry()?;
        Ok(py.allow_threads(|| -> Result<(), VaultError> {
            let _mutation = self.mutation_gate.write().unwrap();
            if self.closed.load(Ordering::Acquire) {
                return Ok(());
            }
            let mut slot = self.conn.lock();
            let conn = slot
                .as_ref()
                .ok_or_else(|| VaultError::Py("KVault is closed".into()))?;
            if self.cache_locked.load(Ordering::Relaxed) {
                return Err(VaultError::Cache("cannot close while cache is locked".into()));
            }
            self.flush_cache_locked(conn)?;
            if !self.read_only {
                let _ = checkpoint_wal(conn);
            }
            self.closed.store(true, Ordering::Release);
            if let Some(readers) = &self.readers {
                readers.close();
            }
            slot.take();
            Ok(())
        })?)
    }

    /// Disable cache (auto-flushes first).
    fn disable_cache(&self, py: Python<'_>) -> PyResult<()> {
        self.check_stream_reentry()?;
        py.allow_threads(|| -> PyResult<()> {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let conn = self.connection()?;
            if self.cache.lock().unwrap().is_none() {
                return Ok(());
            }
            if self.cache_locked.load(Ordering::Relaxed) {
                return Err(VaultError::Cache(
                    "cannot disable cache while flushing is locked".into(),
                )
                .into());
            }
            self.flush_cache_locked(&conn)?;
            *self.cache.lock().unwrap() = None;
            Ok(())
        })
    }

    /// Flush write-back cache (if enabled) in a single transaction.
    /// Respects cache_locked flag - won't flush if locked.
    ///
    /// # Returns
    /// Number of entries flushed
    fn flush_cache(&self, _py: Python<'_>) -> PyResult<usize> {
        _py.allow_threads(|| self.flush_cache_native())
            .map_err(Into::into)
    }

    /// Vacuum & optimize (blocks writer).
    fn optimize(&self, _py: Python<'_>) -> PyResult<()> {
        self.ensure_writable()?;
        _py.allow_threads(|| {
            let conn = self.connection()?;
            conn.execute_batch("PRAGMA optimize; VACUUM;")
                .map_err(VaultError::from)?;
            Ok(())
        })
    }

    /// Get number of keys in the store.
    fn len(&self, py: Python<'_>) -> PyResult<i64> {
        Ok(py.allow_threads(|| {
            let pending: Vec<Vec<u8>> = self
                .cache
                .lock()
                .unwrap()
                .as_ref()
                .map(KvCache::keys)
                .unwrap_or_default();
            self.with_reader(|conn| {
                let mut count: i64 =
                    conn.query_row(&format!("SELECT COUNT(*) FROM {}", self.table), [], |row| {
                        row.get(0)
                    })?;
                let sql = format!("SELECT EXISTS(SELECT 1 FROM {} WHERE key = ?1)", self.table);
                let mut statement = conn.prepare_cached(&sql)?;
                for key in pending {
                    let exists: bool = statement.query_row([key], |row| row.get(0))?;
                    if !exists {
                        count += 1;
                    }
                }
                Ok(count)
            })
        })?)
    }

    /// Native diagnostic snapshot: pending entries, in-flight entries, total payload bytes.
    fn cache_stats(&self, py: Python<'_>) -> (usize, usize, usize) {
        py.allow_threads(|| {
            self.cache
                .lock()
                .unwrap()
                .as_ref()
                .map_or((0, 0, 0), KvCache::stats)
        })
    }

    /// Set cache lock status (for Python lock_cache() context manager).
    fn set_cache_locked(&self, py: Python<'_>, locked: bool) -> PyResult<()> {
        self.check_stream_reentry()?;
        py.allow_threads(|| {
            // Serialize with batch publication; an already active batch may finish.
            let _guard = self.cache.lock().unwrap();
            self.cache_locked.store(locked, Ordering::Relaxed);
        });
        Ok(())
    }

    /// Manually checkpoint WAL file to main database.
    /// This helps prevent WAL from growing too large.
    ///
    /// # Returns
    /// Success indicator (0 on success)
    fn checkpoint_wal(&self, py: Python<'_>) -> PyResult<i64> {
        self.check_stream_reentry()?;
        py.allow_threads(|| {
            let conn = self.connection()?;
            checkpoint_wal(&conn).map_err(|e| e.into())
        })
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

    fn put_many(&self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        self.put_many_impl(py, items)
    }

    #[pyo3(signature = (keys, default=None))]
    fn get_many(
        &self,
        py: Python<'_>,
        keys: &Bound<'_, PyAny>,
        default: Option<PyObject>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_many_impl(py, keys, default)
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.clear_impl(py)
    }

    fn pop(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.pop_impl(py, key)
    }

    #[pyo3(signature = (key, default=None))]
    fn setdefault(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<PyObject>,
    ) -> PyResult<PyObject> {
        self.setdefault_impl(py, key, default.unwrap_or_else(|| py.None()))
    }

    fn reader_pool_stats(&self, py: Python<'_>) -> PyResult<(usize, i64)> {
        self.ensure_open()?;
        Ok(py.allow_threads(|| self.readers.as_ref().map_or((0, 0), ReadPool::stats)))
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

    #[pyo3(signature = (prefix=None, limit=None, after=None))]
    fn scan_keys(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        after: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        self.scan_keys_page(py, prefix, limit, after)
    }

    #[pyo3(signature = (prefix=None, limit=None, after=None))]
    fn scan_items(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        after: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(Py<PyBytes>, PyObject)>> {
        self.scan_items_page(py, prefix, limit, after)
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
        self.ensure_writable()?;
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
    pub(crate) fn connection(
        &self,
    ) -> Result<parking_lot::MappedMutexGuard<'_, Connection>, VaultError> {
        parking_lot::MutexGuard::try_map(self.conn.lock(), |slot| slot.as_mut())
            .map_err(|_| VaultError::Py("KVault is closed".into()))
    }

    pub(crate) fn ensure_open(&self) -> Result<(), VaultError> {
        self.check_stream_reentry()?;
        if self.closed.load(Ordering::Acquire) {
            return Err(VaultError::Py("KVault is closed".into()));
        }
        Ok(())
    }

    pub(crate) fn ensure_writable(&self) -> PyResult<()> {
        self.ensure_open()?;
        if self.read_only {
            return Err(pyo3::exceptions::PyPermissionError::new_err("KVault is read-only"));
        }
        Ok(())
    }

    pub(crate) fn with_point_reader<T>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, VaultError>,
    ) -> Result<T, VaultError> {
        self.ensure_open()?;
        if let Some(readers) = &self.readers {
            readers.query(operation)
        } else {
            let conn = self.connection()?;
            operation(&conn)
        }
    }

    pub(crate) fn with_reader<T>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, VaultError>,
    ) -> Result<T, VaultError> {
        self.ensure_open()?;
        if let Some(readers) = &self.readers {
            readers.read(operation)
        } else {
            let conn = self.connection()?;
            let tx = conn.unchecked_transaction()?;
            let result = operation(&tx);
            tx.rollback()?;
            result
        }
    }

    pub(crate) fn remove_pending(&self, key: &[u8]) {
        if let Some(cache) = self.cache.lock().unwrap().as_mut() {
            cache.remove(key);
        }
    }

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
        self.ensure_writable()?;
        let _mutation = self.mutation_gate.write().unwrap();
        self.ensure_open()?;
        if self.cache_locked.load(Ordering::Relaxed) && self.cache.lock().unwrap().is_some() {
            return Err(
                VaultError::Cache("direct write prohibited while cache is locked".into()).into()
            );
        }
        let conn = self.connection()?;
        conn.prepare_cached(&sql)
            .map_err(VaultError::from)?
            .execute(params![k, v])
            .map_err(VaultError::from)?;
        self.remove_pending(k);
        Ok(())
    }
}

impl _KVault {
    pub(crate) fn flush_cache_native(&self) -> Result<usize, VaultError> {
        self.ensure_open()?;
        // Check if cache is locked (for lock_cache() context manager)
        if self.cache_locked.load(Ordering::Relaxed) {
            return Ok(0); // Skip flush if locked
        }

        let conn = self.connection()?;
        self.ensure_open()?;
        self.flush_cache_batch(&conn, false)
    }

    // Lifecycle callers must never interpret a locked, skipped batch as committed.
    pub(crate) fn flush_cache_locked(&self, conn: &Connection) -> Result<usize, VaultError> {
        self.flush_cache_batch(conn, true)
    }

    // Hold the writer connection, never the cache mutex, through SQL and recovery.
    fn flush_cache_batch(&self, conn: &Connection, required: bool) -> Result<usize, VaultError> {
        let batch = {
            let mut guard = self.cache.lock().unwrap();
            if self.cache_locked.load(Ordering::Relaxed) {
                if required && guard.as_ref().is_some_and(|cache| !cache.is_empty()) {
                    return Err(VaultError::Cache("cannot flush a locked nonempty cache".into()));
                }
                return Ok(0);
            }
            let Some(batch) = guard.as_mut().and_then(KvCache::begin_flush) else {
                return Ok(0);
            };
            batch
        };
        let result = (|| -> Result<usize, VaultError> {
            let sql = format!(
                "INSERT INTO {}(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                self.table
            );
            let tx = conn.unchecked_transaction()?;
            let mut stmt = tx.prepare_cached(&sql)?;
            for (key, value) in &batch.map {
                stmt.execute(params![key, value.as_slice()])?;
            }
            drop(stmt);
            tx.commit()?;
            Ok(batch.map.len())
        })();
        // SQL has committed or rolled back before readers stop seeing the batch.
        drop(batch);
        let retired = self
            .cache
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|cache| cache.finish_flush(result.is_ok()));
        drop(retired);
        result
    }
}
