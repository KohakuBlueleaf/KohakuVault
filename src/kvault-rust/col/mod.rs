// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Columnar storage module for KohakuVault.
//!
//! Provides efficient columnar storage with:
//! - Fixed-size and variable-size columns
//! - Dynamic chunk management with exponential growth
//! - Write-back caching for append operations
//! - Efficient batch operations

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyList};
use rusqlite::{params, Connection};
use thiserror::Error;

// Submodules
pub(crate) mod cache;
pub(crate) mod chunks;
pub(crate) mod fixed;
pub(crate) mod varsize_index;
pub(crate) mod varsize_ops;

// Re-export commonly used items
use cache::ColumnCache;

/// Column operation errors
#[derive(Error, Debug)]
pub enum ColError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Column error: {0}")]
    Col(String),
    #[error("Column not found: {0}")]
    NotFound(String),
    #[error("Cache error: {0}")]
    Cache(String),
}

impl From<ColError> for PyErr {
    fn from(err: ColError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// ColumnVault: Columnar storage implementation for KohakuVault
#[pyclass(name = "_ColumnVault")]
pub struct _ColumnVault {
    pub(crate) conn: Mutex<Connection>,
    /// Per-column caches indexed by col_id
    caches: Mutex<HashMap<i64, ColumnCache>>,
    /// Global cache lock for daemon coordination
    cache_locked: Arc<AtomicBool>,
}

#[pymethods]
impl _ColumnVault {
    /// Create a new ColumnVault using the same database file as KVault.
    ///
    /// Args:
    ///     path: SQLite database file path
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let conn = Connection::open(path).map_err(ColError::from)?;

        // Enable WAL mode for concurrent access
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        // Set WAL auto-checkpoint to 1000 pages (default)
        // This prevents WAL from growing indefinitely
        let _ = conn.pragma_update(None, "wal_autocheckpoint", 1000);
        let _ = conn.pragma_update(None, "synchronous", "NORMAL");

        // CRITICAL: Set busy timeout to prevent infinite hangs when closing
        // If database is locked, SQLite will retry for up to 5 seconds
        conn.busy_timeout(std::time::Duration::from_millis(5000))
            .map_err(ColError::from)?;

        // Create schema for columnar storage
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS kohakuvault_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS col_meta (
                col_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                dtype TEXT NOT NULL,
                elem_size INTEGER NOT NULL,
                length INTEGER NOT NULL,
                chunk_bytes INTEGER NOT NULL,
                min_chunk_bytes INTEGER DEFAULT 131072,
                max_chunk_bytes INTEGER DEFAULT 16777216
            );

            CREATE INDEX IF NOT EXISTS col_meta_name_idx ON col_meta(name);

            CREATE TABLE IF NOT EXISTS col_chunks (
                col_id INTEGER NOT NULL,
                chunk_idx INTEGER NOT NULL,
                data BLOB NOT NULL,
                actual_size INTEGER NOT NULL,
                bytes_used INTEGER NOT NULL DEFAULT 0,
                has_deleted INTEGER NOT NULL DEFAULT 0,
                start_elem_idx INTEGER DEFAULT 0,
                end_elem_idx INTEGER DEFAULT 0,
                PRIMARY KEY (col_id, chunk_idx),
                FOREIGN KEY (col_id) REFERENCES col_meta(col_id) ON DELETE CASCADE
            );
            ",
        )
        .map_err(ColError::from)?;

        // Set schema version for new databases
        let version_exists: Result<String, _> = conn.query_row(
            "SELECT value FROM kohakuvault_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        );

        if version_exists.is_err() {
            // New database - set schema version to 2 (no-cross-chunk)
            conn.execute(
                "INSERT INTO kohakuvault_meta (key, value) VALUES ('schema_version', '2')",
                [],
            )
            .map_err(ColError::from)?;
        }

        Ok(Self {
            conn: Mutex::new(conn),
            caches: Mutex::new(HashMap::new()),
            cache_locked: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a new column with given name, dtype, and chunk sizes.
    ///
    /// Args:
    ///     name: Column name (must be unique)
    ///     dtype: Data type string ("i64", "f64", "bytes:N", "bytes")
    ///     elem_size: Size of each element in bytes
    ///     min_chunk_bytes: Minimum chunk size (default 128KB)
    ///     max_chunk_bytes: Maximum chunk size (default 16MB)
    ///
    /// Returns:
    ///     col_id: Integer ID of created column
    #[pyo3(signature = (name, dtype, elem_size, min_chunk_bytes=131072, max_chunk_bytes=16777216))]
    fn create_column(
        &self,
        name: &str,
        dtype: &str,
        elem_size: i64,
        min_chunk_bytes: i64,
        max_chunk_bytes: i64,
    ) -> PyResult<i64> {
        // Align chunk sizes to element size for fixed-size columns
        let (aligned_min, aligned_max) = if elem_size > 1 {
            chunks::align_chunk_sizes(elem_size, min_chunk_bytes, max_chunk_bytes)?
        } else {
            // elem_size=1 (bytes) - no alignment needed
            (min_chunk_bytes, max_chunk_bytes)
        };

        let conn = self.conn.lock().unwrap();

        conn.execute(
            "
            INSERT INTO col_meta (name, dtype, elem_size, length, chunk_bytes, min_chunk_bytes, max_chunk_bytes)
            VALUES (?1, ?2, ?3, 0, ?4, ?5, ?6)
            ",
            params![name, dtype, elem_size, aligned_min, aligned_min, aligned_max],
        )
        .map_err(ColError::from)?;

        let col_id = conn.last_insert_rowid();
        Ok(col_id)
    }

    // ========================================
    // CACHE METHODS
    // ========================================

    /// Enable cache for a specific column.
    ///
    /// Args:
    ///     col_id: Column ID (data_col_id for variable-size columns)
    ///     cap_bytes: Maximum cache size in bytes
    ///     flush_threshold: Auto-flush trigger point
    ///     is_variable_size: Whether this is a variable-size column
    ///     idx_col_id: For variable-size columns, the index column ID (optional)
    #[pyo3(signature = (col_id, cap_bytes, flush_threshold, is_variable_size, idx_col_id=None))]
    fn enable_cache(
        &self,
        col_id: i64,
        cap_bytes: usize,
        flush_threshold: usize,
        is_variable_size: bool,
        idx_col_id: Option<i64>,
    ) -> PyResult<()> {
        let mut caches = self.caches.lock().unwrap();
        caches.insert(
            col_id,
            ColumnCache::new(cap_bytes, flush_threshold, is_variable_size, idx_col_id),
        );
        Ok(())
    }

    /// Disable cache for a specific column (auto-flushes first).
    fn disable_cache(&self, col_id: i64) -> PyResult<()> {
        // Flush before disabling
        self.flush_cache(col_id)?;

        let mut caches = self.caches.lock().unwrap();
        caches.remove(&col_id);
        Ok(())
    }

    /// Check if cache is enabled for a column.
    fn is_cache_enabled(&self, col_id: i64) -> bool {
        let caches = self.caches.lock().unwrap();
        caches.contains_key(&col_id)
    }

    /// Append data to cache (fixed-size columns).
    /// Returns true if cache was auto-flushed due to threshold.
    fn append_cached(
        &self,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        _elem_size: i64,
        _current_length: i64,
    ) -> PyResult<bool> {
        let data_vec = data.as_bytes().to_vec();
        let data_size = data_vec.len();

        let mut caches = self.caches.lock().unwrap();
        let cache = caches
            .get_mut(&col_id)
            .ok_or_else(|| ColError::Cache(format!("Cache not enabled for col_id {}", col_id)))?;

        // Check if adding this data would exceed capacity
        if cache.would_exceed_capacity(data_size) {
            // Auto-flush to make room
            drop(caches);
            self.flush_cache(col_id)?;

            // Reacquire lock and add to now-empty cache
            let mut caches = self.caches.lock().unwrap();
            let cache = caches.get_mut(&col_id).unwrap();
            cache.append(data_vec);
            return Ok(true);
        }

        let needs_flush = cache.append(data_vec);

        if needs_flush {
            // Drop the lock before flushing to avoid deadlock
            drop(caches);
            self.flush_cache(col_id)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Extend cache with multiple elements (fixed-size or variable-size).
    /// For variable-size, values is a list of PyBytes.
    /// Returns true if cache was auto-flushed.
    fn extend_cached(
        &self,
        col_id: i64,
        values: &Bound<'_, PyList>,
        is_variable_size: bool,
    ) -> PyResult<bool> {
        let elements: Vec<Vec<u8>> = if is_variable_size {
            values
                .iter()
                .map(|v| v.downcast::<PyBytes>().unwrap().as_bytes().to_vec())
                .collect()
        } else {
            values
                .iter()
                .map(|v| v.downcast::<PyBytes>().unwrap().as_bytes().to_vec())
                .collect()
        };

        // Calculate total size of elements
        let total_size: usize = elements.iter().map(|e| e.len()).sum();

        let mut caches = self.caches.lock().unwrap();
        let cache = caches
            .get_mut(&col_id)
            .ok_or_else(|| ColError::Cache(format!("Cache not enabled for col_id {}", col_id)))?;

        // Check if adding all elements would exceed capacity
        if cache.would_exceed_capacity(total_size) {
            // Auto-flush to make room
            drop(caches);
            self.flush_cache(col_id)?;

            // Reacquire lock and add to now-empty cache
            let mut caches = self.caches.lock().unwrap();
            let cache = caches.get_mut(&col_id).unwrap();
            cache.extend(elements);
            return Ok(true);
        }

        let needs_flush = cache.extend(elements);

        if needs_flush {
            drop(caches);
            self.flush_cache(col_id)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Flush cache for a specific column.
    /// Returns number of bytes flushed.
    fn flush_cache(&self, col_id: i64) -> PyResult<usize> {
        let mut caches = self.caches.lock().unwrap();

        let cache = match caches.get_mut(&col_id) {
            Some(c) => c,
            None => return Ok(0), // No cache enabled, nothing to flush
        };

        if cache.is_empty() {
            return Ok(0);
        }

        let (fixed_buffer, var_buffer) = cache.take();
        let is_variable_size = cache.is_variable_size;
        let idx_col_id = cache.idx_col_id;

        let bytes_flushed = if is_variable_size {
            var_buffer.iter().map(|v| v.len()).sum()
        } else {
            fixed_buffer.len()
        };

        // Drop the lock before performing I/O
        drop(caches);

        // Get column metadata
        let conn = self.conn.lock().unwrap();
        let (elem_size, current_length): (i64, i64) = conn
            .query_row(
                "SELECT elem_size, length FROM col_meta WHERE col_id = ?1",
                params![col_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ColError::from)?;
        drop(conn);

        // Perform the flush using existing append/extend methods
        if fixed_buffer.is_empty() && var_buffer.is_empty() {
            return Ok(0);
        }

        if !fixed_buffer.is_empty() {
            // Fixed-size column: use append_raw
            Python::with_gil(|py| {
                let py_bytes = PyBytes::new_bound(py, &fixed_buffer);
                self.append_raw_impl(col_id, &py_bytes, elem_size, 0, current_length)
            })?;
        } else if !var_buffer.is_empty() {
            // Variable-size column: use extend_adaptive
            // Get max_chunk_bytes from metadata first (outside with_gil)
            let conn = self.conn.lock().unwrap();
            let max_chunk_bytes: i64 = conn
                .query_row(
                    "SELECT max_chunk_bytes FROM col_meta WHERE col_id = ?1",
                    params![col_id],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;
            drop(conn);

            // Call extend_adaptive and get index data back
            let index_bytes = Python::with_gil(|py| {
                let py_list = PyList::new_bound(
                    py,
                    var_buffer
                        .iter()
                        .map(|v| PyBytes::new_bound(py, v))
                        .collect::<Vec<_>>(),
                );

                self.extend_adaptive_impl(py, col_id, &py_list, max_chunk_bytes)
            })?;

            // Now append the index data to the index column
            if let Some(idx_col_id) = idx_col_id {
                let conn = self.conn.lock().unwrap();
                let (_, idx_length): (i64, i64) = conn
                    .query_row(
                        "SELECT elem_size, length FROM col_meta WHERE col_id = ?1",
                        params![idx_col_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .map_err(ColError::from)?;
                drop(conn);

                Python::with_gil(|py| {
                    let idx_bytes = index_bytes.bind(py);
                    self.append_raw_impl(idx_col_id, idx_bytes, 12, 0, idx_length)
                })?;
            }
        }

        Ok(bytes_flushed)
    }

    /// Flush all column caches.
    /// Returns total number of bytes flushed.
    fn flush_all_caches(&self) -> PyResult<usize> {
        let caches = self.caches.lock().unwrap();
        let col_ids: Vec<i64> = caches.keys().copied().collect();
        drop(caches);

        let mut total_flushed = 0;
        for col_id in col_ids {
            total_flushed += self.flush_cache(col_id)?;
        }

        Ok(total_flushed)
    }

    /// Lock cache to prevent daemon flushes.
    fn lock_cache(&self) {
        self.cache_locked.store(true, Ordering::SeqCst);
    }

    /// Unlock cache to allow daemon flushes.
    fn unlock_cache(&self) {
        self.cache_locked.store(false, Ordering::SeqCst);
    }

    /// Check if cache is locked.
    fn is_cache_locked(&self) -> bool {
        self.cache_locked.load(Ordering::SeqCst)
    }

    /// Get idle time for a column's cache (in seconds).
    /// Returns None if cache not enabled.
    fn get_cache_idle_time(&self, col_id: i64) -> Option<f64> {
        let caches = self.caches.lock().unwrap();
        caches.get(&col_id).map(|c| c.idle_time().as_secs_f64())
    }

    // ========================================
    // END CACHE METHODS
    // ========================================

    /// Get column metadata by name.
    ///
    /// Returns:
    ///     (col_id, elem_size, length, max_chunk_bytes)
    fn get_column_info(&self, py: Python<'_>, name: &str) -> PyResult<(i64, i64, i64, i64)> {
        let name = name.to_string();

        py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();

            let result = conn.query_row(
                "
                SELECT col_id, elem_size, length, max_chunk_bytes
                FROM col_meta
                WHERE name = ?1
                ",
                params![name],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            );

            match result {
                Ok(info) => Ok(info),
                Err(rusqlite::Error::QueryReturnedNoRows) => Err(ColError::NotFound(name).into()),
                Err(e) => Err(ColError::from(e).into()),
            }
        })
    }

    /// List all columns with their metadata.
    ///
    /// Returns:
    ///     List of (name, dtype, length) tuples
    fn list_columns(&self, py: Python<'_>) -> PyResult<Vec<(String, String, i64)>> {
        py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();

            let mut stmt = conn
                .prepare("SELECT name, dtype, length FROM col_meta ORDER BY col_id")
                .map_err(ColError::from)?;

            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?))
                })
                .map_err(ColError::from)?;

            let mut result = Vec::new();
            for row in rows {
                result.push(row.map_err(ColError::from)?);
            }

            Ok(result)
        })
    }

    /// Delete a column and all its data.
    fn delete_column(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let name = name.to_string();

        py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();

            let deleted = conn
                .execute("DELETE FROM col_meta WHERE name = ?1", params![name])
                .map_err(ColError::from)?;

            Ok(deleted > 0)
        })
    }

    /// Manually checkpoint WAL file to main database.
    /// This helps prevent WAL from growing too large.
    /// Returns success indicator (0 = success).
    fn checkpoint_wal(&self, py: Python<'_>) -> PyResult<i64> {
        py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();
            // PRAGMA wal_checkpoint(PASSIVE) - non-blocking checkpoint
            conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")
                .map_err(ColError::from)?;
            Ok(0) // Success indicator
        })
    }

    // ===== Fixed-Size Column Operations =====

    fn read_range(
        &self,
        py: Python<'_>,
        col_id: i64,
        start_idx: i64,
        count: i64,
        elem_size: i64,
        chunk_bytes: i64,
    ) -> PyResult<Py<PyBytes>> {
        self.read_range_impl(py, col_id, start_idx, count, elem_size, chunk_bytes)
    }

    fn write_range(
        &self,
        col_id: i64,
        start_idx: i64,
        data: &Bound<'_, PyBytes>,
        elem_size: i64,
        chunk_bytes: i64,
    ) -> PyResult<()> {
        self.write_range_impl(col_id, start_idx, data, elem_size, chunk_bytes)
    }

    #[pyo3(signature = (col_id, start_idx, count, elem_size, chunk_bytes, packer=None))]
    #[allow(clippy::too_many_arguments)]
    fn batch_read_fixed(
        &self,
        py: Python<'_>,
        col_id: i64,
        start_idx: i64,
        count: i64,
        elem_size: i64,
        chunk_bytes: i64,
        packer: Option<&crate::packer::DataPacker>,
    ) -> PyResult<Py<PyList>> {
        self.batch_read_fixed_impl(py, col_id, start_idx, count, elem_size, chunk_bytes, packer)
    }

    fn append_raw(
        &self,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        elem_size: i64,
        min_chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        self.append_raw_impl(col_id, data, elem_size, min_chunk_bytes, current_length)
    }

    fn set_length(&self, col_id: i64, new_length: i64) -> PyResult<()> {
        self.set_length_impl(col_id, new_length)
    }

    fn append_typed(
        &self,
        col_id: i64,
        value: &Bound<'_, PyAny>,
        packer: &crate::packer::DataPacker,
        chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        self.append_typed_impl(col_id, value, packer, chunk_bytes, current_length)
    }

    fn extend_typed(
        &self,
        col_id: i64,
        values: &Bound<'_, PyList>,
        packer: &crate::packer::DataPacker,
        chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        self.extend_typed_impl(col_id, values, packer, chunk_bytes, current_length)
    }

    fn append_typed_cached(
        &self,
        col_id: i64,
        value: &Bound<'_, PyAny>,
        packer: &crate::packer::DataPacker,
        current_length: i64,
    ) -> PyResult<bool> {
        self.append_typed_cached_impl(col_id, value, packer, current_length)
    }

    fn extend_typed_cached(
        &self,
        col_id: i64,
        values: &Bound<'_, PyList>,
        packer: &crate::packer::DataPacker,
    ) -> PyResult<bool> {
        self.extend_typed_cached_impl(col_id, values, packer)
    }

    // ===== Variable-Size Column Operations =====

    fn read_adaptive(
        &self,
        py: Python<'_>,
        col_id: i64,
        chunk_id: i32,
        start_byte: i32,
        end_byte: i32,
    ) -> PyResult<Py<PyBytes>> {
        self.read_adaptive_impl(py, col_id, chunk_id, start_byte, end_byte)
    }

    #[pyo3(signature = (idx_col_id, data_col_id, start_idx, count, idx_elem_size, idx_chunk_bytes))]
    #[allow(clippy::too_many_arguments)]
    fn batch_read_varsize(
        &self,
        py: Python<'_>,
        idx_col_id: i64,
        data_col_id: i64,
        start_idx: i64,
        count: i64,
        idx_elem_size: i64,
        idx_chunk_bytes: i64,
    ) -> PyResult<Py<PyList>> {
        self.batch_read_varsize_impl(
            py,
            idx_col_id,
            data_col_id,
            start_idx,
            count,
            idx_elem_size,
            idx_chunk_bytes,
        )
    }

    #[pyo3(signature = (idx_col_id, data_col_id, start_idx, count, idx_elem_size, idx_chunk_bytes, packer))]
    #[allow(clippy::too_many_arguments)]
    fn batch_read_varsize_unpacked(
        &self,
        py: Python<'_>,
        idx_col_id: i64,
        data_col_id: i64,
        start_idx: i64,
        count: i64,
        idx_elem_size: i64,
        idx_chunk_bytes: i64,
        packer: &crate::packer::DataPacker,
    ) -> PyResult<Py<PyList>> {
        self.batch_read_varsize_unpacked_impl(
            py,
            idx_col_id,
            data_col_id,
            start_idx,
            count,
            idx_elem_size,
            idx_chunk_bytes,
            packer,
        )
    }

    fn append_raw_adaptive(
        &self,
        py: Python<'_>,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        max_chunk_bytes: i64,
    ) -> PyResult<(i32, i32, i32)> {
        self.append_raw_adaptive_impl(py, col_id, data, max_chunk_bytes)
    }

    fn extend_adaptive(
        &self,
        py: Python<'_>,
        col_id: i64,
        data_list: &Bound<'_, PyList>,
        max_chunk_bytes: i64,
    ) -> PyResult<Py<PyBytes>> {
        self.extend_adaptive_impl(py, col_id, data_list, max_chunk_bytes)
    }

    fn append_raw_adaptive_cached(&self, col_id: i64, data: &Bound<'_, PyBytes>) -> PyResult<bool> {
        self.append_raw_adaptive_cached_impl(col_id, data)
    }

    fn extend_adaptive_cached(&self, col_id: i64, data_list: &Bound<'_, PyList>) -> PyResult<bool> {
        self.extend_adaptive_cached_impl(col_id, data_list)
    }

    // ===== Variable-Size Index Operations =====

    #[allow(clippy::too_many_arguments)]
    fn update_varsize_element(
        &self,
        data_col_id: i64,
        idx_col_id: i64,
        elem_idx: i64,
        new_data: &Bound<'_, PyBytes>,
        chunk_id: i32,
        old_start: i32,
        old_end: i32,
        max_chunk_bytes: i64,
    ) -> PyResult<(i32, i32, i32)> {
        Python::with_gil(|py| {
            self.update_varsize_element_impl(
                py,
                data_col_id,
                idx_col_id,
                elem_idx,
                new_data,
                chunk_id,
                old_start,
                old_end,
                max_chunk_bytes,
            )
        })
    }

    fn update_varsize_slice(
        &self,
        data_col_id: i64,
        idx_col_id: i64,
        start_idx: i64,
        count: i64,
        new_values: &Bound<'_, PyList>,
        max_chunk_bytes: i64,
    ) -> PyResult<()> {
        Python::with_gil(|py| {
            self.update_varsize_slice_impl(
                py,
                data_col_id,
                idx_col_id,
                start_idx,
                count,
                new_values,
                max_chunk_bytes,
            )
        })
    }

    fn delete_adaptive(&self, idx_col_id: i64, elem_idx: i64) -> PyResult<i32> {
        self.delete_adaptive_impl(idx_col_id, elem_idx)
    }
}
