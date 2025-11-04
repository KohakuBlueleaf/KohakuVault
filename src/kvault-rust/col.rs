// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyList};
use rusqlite::{params, Connection};
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ColError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Column error: {0}")]
    Col(String),
    #[error("Column not found: {0}")]
    NotFound(String),
}

impl From<ColError> for PyErr {
    fn from(e: ColError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

/// Internal columnar storage implementation.
/// Python wrapper provides list-like interface and type handling.
#[pyclass]
pub struct _ColumnVault {
    conn: Mutex<Connection>,
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

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Create a new column with given name, dtype, and chunk sizes.
    ///
    /// Args:
    ///     name: Column name (must be unique)
    ///     dtype: Data type string ("i64", "f64", "bytes:N", "bytes")
    ///     elem_size: Size of each element in bytes
    ///     chunk_bytes: Initial/default chunk size (compatibility)
    ///     min_chunk_bytes: Minimum chunk size (default 128KB)
    ///     max_chunk_bytes: Maximum chunk size (default 16MB)
    ///
    /// Returns:
    ///     col_id: Integer ID of created column
    #[pyo3(signature = (name, dtype, elem_size, chunk_bytes, min_chunk_bytes=131072, max_chunk_bytes=16777216))]
    fn create_column(
        &self,
        name: &str,
        dtype: &str,
        elem_size: i64,
        chunk_bytes: i64,
        min_chunk_bytes: i64,
        max_chunk_bytes: i64,
    ) -> PyResult<i64> {
        // Align chunk sizes to element size for fixed-size columns
        let (aligned_min, aligned_max) = if elem_size > 1 {
            Self::align_chunk_sizes(elem_size, min_chunk_bytes, max_chunk_bytes)?
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
            params![name, dtype, elem_size, chunk_bytes, aligned_min, aligned_max],
        )
        .map_err(ColError::from)?;

        let col_id = conn.last_insert_rowid();
        Ok(col_id)
    }

    /// Get column metadata by name.
    ///
    /// Returns:
    ///     (col_id, elem_size, length, max_chunk_bytes)
    fn get_column_info(&self, name: &str) -> PyResult<(i64, i64, i64, i64)> {
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
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(ColError::NotFound(name.to_string()).into())
            }
            Err(e) => Err(ColError::from(e).into()),
        }
    }

    /// Read a range of elements from a column (returns raw bytes).
    ///
    /// Args:
    ///     col_id: Column ID
    ///     start_idx: Starting element index
    ///     count: Number of elements to read
    ///     elem_size: Size of each element in bytes
    ///     chunk_bytes: Chunk size in bytes
    ///
    /// Returns:
    ///     Raw bytes containing packed elements
    fn read_range(
        &self,
        py: Python<'_>,
        col_id: i64,
        start_idx: i64,
        count: i64,
        elem_size: i64,
        chunk_bytes: i64,
    ) -> PyResult<Py<PyBytes>> {
        let conn = self.conn.lock().unwrap();

        // With element-aligned chunks: elements don't cross chunks,
        // but a read_range can span multiple chunks
        let start_byte = start_idx * elem_size;
        let total_bytes = (count * elem_size) as usize;
        let end_byte = start_byte + total_bytes as i64;

        // Calculate chunk range
        let start_chunk = start_byte / chunk_bytes;
        let end_chunk = (end_byte - 1) / chunk_bytes;

        let mut result = vec![0u8; total_bytes];
        let mut result_offset = 0;

        // Read from each chunk (simplified - no partial element handling)
        for chunk_idx in start_chunk..=end_chunk {
            let chunk_start_byte = chunk_idx * chunk_bytes;
            let chunk_end_byte = chunk_start_byte + chunk_bytes;

            let read_start = std::cmp::max(start_byte, chunk_start_byte);
            let read_end = std::cmp::min(end_byte, chunk_end_byte);
            let bytes_to_read = (read_end - read_start) as usize;

            if bytes_to_read == 0 {
                continue;
            }

            let offset_in_chunk = (read_start - chunk_start_byte) as usize;

            let chunk_data: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            result[result_offset..result_offset + bytes_to_read]
                .copy_from_slice(&chunk_data[offset_in_chunk..offset_in_chunk + bytes_to_read]);

            result_offset += bytes_to_read;
        }

        Ok(PyBytes::new_bound(py, &result).unbind())
    }

    /// Read from adaptive variable-size storage (single chunk, known offsets)
    fn read_adaptive(
        &self,
        py: Python<'_>,
        col_id: i64,
        chunk_id: i32,
        start_byte: i32,
        end_byte: i32,
    ) -> PyResult<Py<PyBytes>> {
        let conn = self.conn.lock().unwrap();

        // Get rowid for BLOB read
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_id as i64],
                |row| row.get(0),
            )
            .map_err(|e| {
                ColError::Col(format!(
                    "Chunk not found: col_id={}, chunk_id={}, error={}",
                    col_id, chunk_id, e
                ))
            })?;

        // Use BLOB API for efficient read
        let blob = conn
            .blob_open(
                rusqlite::DatabaseName::Main,
                "col_chunks",
                "data",
                rowid,
                true, // read_only=true means READ ONLY
            )
            .map_err(ColError::from)?;

        let len = (end_byte - start_byte) as usize;
        let mut result = vec![0u8; len];
        blob.read_at(&mut result, start_byte as usize)
            .map_err(ColError::from)?;

        Ok(PyBytes::new_bound(py, &result).unbind())
    }

    /// Write a range of elements to a column (from raw bytes).
    ///
    /// Args:
    ///     col_id: Column ID
    ///     start_idx: Starting element index
    ///     data: Raw bytes to write
    ///     elem_size: Size of each element in bytes
    ///     chunk_bytes: Chunk size in bytes
    fn write_range(
        &self,
        col_id: i64,
        start_idx: i64,
        data: &Bound<'_, PyBytes>,
        elem_size: i64,
        chunk_bytes: i64,
    ) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        let data_bytes = data.as_bytes();

        // With element-aligned chunks: write can span multiple chunks
        let start_byte = start_idx * elem_size;
        let total_bytes = data_bytes.len();
        let end_byte = start_byte + total_bytes as i64;

        // Calculate chunk range
        let start_chunk = start_byte / chunk_bytes;
        let end_chunk = (end_byte - 1) / chunk_bytes;

        let mut data_offset = 0;

        // Write to each chunk (simplified - no partial element handling)
        for chunk_idx in start_chunk..=end_chunk {
            let chunk_start_byte = chunk_idx * chunk_bytes;
            let chunk_end_byte = chunk_start_byte + chunk_bytes;

            let write_start = std::cmp::max(start_byte, chunk_start_byte);
            let write_end = std::cmp::min(end_byte, chunk_end_byte);
            let bytes_to_write = (write_end - write_start) as usize;

            if bytes_to_write == 0 {
                continue;
            }

            let offset_in_chunk = (write_start - chunk_start_byte) as usize;

            // Ensure chunk exists
            self.ensure_chunk(&conn, col_id, chunk_idx, false)?;

            // Update chunk data
            let mut chunk_data: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            chunk_data[offset_in_chunk..offset_in_chunk + bytes_to_write]
                .copy_from_slice(&data_bytes[data_offset..data_offset + bytes_to_write]);

            conn.execute(
                "UPDATE col_chunks SET data = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                params![chunk_data, col_id, chunk_idx],
            )
            .map_err(ColError::from)?;

            data_offset += bytes_to_write;
        }

        Ok(())
    }

    /// Append raw bytes to the end of a column with dynamic chunk growth.
    /// Most performance-critical operation.
    ///
    /// Strategy for large appends that span multiple chunks:
    /// 1. Fill current last chunk (grow to max if needed)
    /// 2. Create new chunks with exponential growth: 2^k * min_chunk_bytes
    /// 3. Each new chunk size is capped by: min(max_chunk_bytes, remaining_data_size)
    /// 4. Maintains invariant: all chunks 0..k-1 are at max_chunk_bytes
    ///
    /// Args:
    ///     col_id: Column ID
    ///     data: Raw bytes to append
    ///     elem_size: Size of each element (for elem_size=1, treats as raw bytes)
    ///     chunk_bytes: Unused (kept for API compatibility)
    ///     current_length: Current number of elements (for elem_size>1) or bytes (elem_size=1)
    fn append_raw(
        &self,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        elem_size: i64,
        _chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        let data_bytes = data.as_bytes();
        let total_bytes_to_append = data_bytes.len();

        // For raw bytes (elem_size=1), current_length is in bytes
        // For typed data, current_length is in elements
        let current_byte_offset = if elem_size == 1 {
            current_length
        } else {
            current_length * elem_size
        };

        // Get chunk size settings
        let (min_chunk, max_chunk): (i64, i64) = conn
            .query_row(
                "SELECT min_chunk_bytes, max_chunk_bytes FROM col_meta WHERE col_id = ?1",
                params![col_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ColError::from)?;

        let mut bytes_written = 0;
        let mut current_offset = current_byte_offset;

        while bytes_written < total_bytes_to_append {
            let remaining_bytes = total_bytes_to_append - bytes_written;

            // Get or create chunk to write to
            let (chunk_idx, offset_in_chunk, chunk_capacity) = self.prepare_append_chunk(
                &conn,
                col_id,
                current_offset,
                remaining_bytes,
                min_chunk,
                max_chunk,
            )?;

            // Calculate how much to write to this chunk
            let space_available = chunk_capacity - offset_in_chunk;
            let bytes_to_write = std::cmp::min(remaining_bytes, space_available);

            // Get rowid for incremental blob I/O
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Use incremental BLOB I/O to write directly without reading entire blob
            // CRITICAL: Last parameter must be TRUE for write access!
            let mut blob = conn
                .blob_open(rusqlite::DatabaseName::Main, "col_chunks", "data", rowid, false)
                .map_err(ColError::from)?;

            blob.write_at(
                &data_bytes[bytes_written..bytes_written + bytes_to_write],
                offset_in_chunk,
            )
            .map_err(ColError::from)?;

            bytes_written += bytes_to_write;
            current_offset += bytes_to_write as i64;
        }

        // Update length
        let new_length = if elem_size == 1 {
            current_length + total_bytes_to_append as i64
        } else {
            current_length + (total_bytes_to_append as i64 / elem_size)
        };

        conn.execute(
            "UPDATE col_meta SET length = ?1 WHERE col_id = ?2",
            params![new_length, col_id],
        )
        .map_err(ColError::from)?;

        Ok(())
    }

    /// Update the length of a column in metadata.
    fn set_length(&self, col_id: i64, new_length: i64) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE col_meta SET length = ?1 WHERE col_id = ?2",
            params![new_length, col_id],
        )
        .map_err(ColError::from)?;

        Ok(())
    }

    /// List all columns with their metadata.
    ///
    /// Returns:
    ///     List of (name, dtype, length) tuples
    fn list_columns(&self, _py: Python<'_>) -> PyResult<Vec<(String, String, i64)>> {
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
    }

    /// Delete a column and all its data.
    fn delete_column(&self, name: &str) -> PyResult<bool> {
        let conn = self.conn.lock().unwrap();

        let deleted = conn
            .execute("DELETE FROM col_meta WHERE name = ?1", params![name])
            .map_err(ColError::from)?;

        Ok(deleted > 0)
    }

    /// Append value using DataPacker (NEW - typed interface)
    fn append_typed(
        &self,
        col_id: i64,
        value: &Bound<'_, PyAny>,
        packer: &crate::packer::DataPacker,
        chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        // Pack value in Rust
        let packed_bytes = packer.pack(value.py(), value)?;
        let py_bytes = packed_bytes.bind(value.py());

        // Get element size from packer
        let elem_size = packer.elem_size() as i64;

        // Append raw bytes (existing method)
        self.append_raw(col_id, py_bytes, elem_size, chunk_bytes, current_length)
    }

    /// Extend with multiple values using DataPacker (NEW - typed interface)
    fn extend_typed(
        &self,
        col_id: i64,
        values: &Bound<'_, PyList>,
        packer: &crate::packer::DataPacker,
        chunk_bytes: i64,
        current_length: i64,
    ) -> PyResult<()> {
        // Pack all values in Rust (single concatenated bytes)
        let packed_bytes = packer.pack_many(values.py(), values)?;
        let py_bytes = packed_bytes.bind(values.py());

        let elem_size = packer.elem_size() as i64;

        // Append all at once (existing method)
        self.append_raw(col_id, py_bytes, elem_size, chunk_bytes, current_length)
    }

    /// Append raw data to variable-size column with adaptive chunking.
    /// Returns (chunk_id, start_byte, end_byte) as i32 triple.
    ///
    /// New strategy (v0.4.0):
    /// - Tracks bytes_used vs chunk_size (capacity)
    /// - Only expands when truly needed
    /// - Smart expansion based on element size
    fn append_raw_adaptive(
        &self,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        max_chunk_bytes: i64,
    ) -> PyResult<(i32, i32, i32)> {
        let data_bytes = data.as_bytes().to_vec();
        let needed = data_bytes.len() as i64;

        let conn = self.conn.lock().unwrap();

        // Get min_chunk_bytes
        let min_chunk_bytes: i64 = conn
            .query_row(
                "SELECT min_chunk_bytes FROM col_meta WHERE col_id = ?1",
                params![col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Special case: element > max_chunk_bytes
        // Create dedicated chunk of exact size
        if needed > max_chunk_bytes {
            let next_id = match conn.query_row(
                "SELECT MAX(chunk_idx) FROM col_chunks WHERE col_id = ?1",
                params![col_id],
                |row| row.get::<_, Option<i64>>(0),
            ) {
                Ok(Some(max_idx)) => max_idx + 1,
                _ => 0,
            };

            conn.execute(
                "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size, bytes_used)
                 VALUES (?1, ?2, zeroblob(?3), ?3, ?4)",
                params![col_id, next_id, needed, needed],
            )
            .map_err(ColError::from)?;

            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, next_id],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            let mut blob = conn
                .blob_open(rusqlite::DatabaseName::Main, "col_chunks", "data", rowid, false)
                .map_err(ColError::from)?;

            blob.write_at(&data_bytes, 0).map_err(ColError::from)?;

            return Ok((next_id as i32, 0, needed as i32));
        }

        // Query last chunk with bytes_used tracking
        let (chunk_id, _chunk_size, bytes_used) = match conn.query_row(
            "SELECT chunk_idx, actual_size, bytes_used FROM col_chunks
             WHERE col_id = ?1 ORDER BY chunk_idx DESC LIMIT 1",
            params![col_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        ) {
            Ok((last_id, chunk_size, bytes_used)) => {
                let available = chunk_size - bytes_used;

                // CASE 1: Fits without expansion
                if available >= needed {
                    (last_id, chunk_size, bytes_used)
                }
                // CASE 2: max - used >= needed (can expand to legal size)
                else if max_chunk_bytes - bytes_used >= needed {
                    // Find legal_size = min * 2^k where legal_size - used >= needed
                    let target = bytes_used + needed;
                    let mut legal_size = min_chunk_bytes;
                    while legal_size < target && legal_size < max_chunk_bytes {
                        legal_size *= 2;
                    }
                    legal_size = std::cmp::min(legal_size, max_chunk_bytes);

                    // Expand to legal_size
                    conn.execute(
                        "UPDATE col_chunks
                         SET data = data || zeroblob(?1 - length(data)),
                             actual_size = ?1
                         WHERE col_id = ?2 AND chunk_idx = ?3",
                        params![legal_size, col_id, last_id],
                    )
                    .map_err(ColError::from)?;

                    (last_id, legal_size, bytes_used)
                }
                // CASE 3: max - used < needed
                else if chunk_size < max_chunk_bytes && needed <= max_chunk_bytes {
                    // 3-1: Not at max yet, expand to fit
                    let new_size = bytes_used + needed;
                    conn.execute(
                        "UPDATE col_chunks
                         SET data = data || zeroblob(?1 - length(data)),
                             actual_size = ?1
                         WHERE col_id = ?2 AND chunk_idx = ?3",
                        params![new_size, col_id, last_id],
                    )
                    .map_err(ColError::from)?;

                    (last_id, new_size, bytes_used)
                } else if chunk_size >= max_chunk_bytes && needed <= max_chunk_bytes / 2 {
                    // 3-2: At max, small element - expand to 1.5x
                    let new_size = bytes_used + needed;
                    conn.execute(
                        "UPDATE col_chunks
                         SET data = data || zeroblob(?1 - length(data)),
                             actual_size = ?1
                         WHERE col_id = ?2 AND chunk_idx = ?3",
                        params![new_size, col_id, last_id],
                    )
                    .map_err(ColError::from)?;

                    (last_id, new_size, bytes_used)
                } else {
                    // 3-3: At max, large element - create new chunk
                    let mut size = min_chunk_bytes;
                    while size < needed && size < max_chunk_bytes {
                        size *= 2;
                    }
                    size = std::cmp::min(size, max_chunk_bytes);

                    conn.execute(
                        "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size, bytes_used)
                         VALUES (?1, ?2, zeroblob(?3), ?3, 0)",
                        params![col_id, last_id + 1, size],
                    )
                    .map_err(ColError::from)?;

                    (last_id + 1, size, 0)
                }
            }
            Err(_) => {
                // No chunks exist - create first one
                let mut size = min_chunk_bytes;
                while size < needed && size < max_chunk_bytes {
                    size *= 2;
                }
                size = std::cmp::min(size, max_chunk_bytes);

                conn.execute(
                    "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size, bytes_used)
                     VALUES (?1, ?2, zeroblob(?3), ?3, 0)",
                    params![col_id, 0, size],
                )
                .map_err(ColError::from)?;

                (0, size, 0)
            }
        };

        // Write data using BLOB incremental I/O
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        let mut blob = conn
            .blob_open(rusqlite::DatabaseName::Main, "col_chunks", "data", rowid, false)
            .map_err(ColError::from)?;

        blob.write_at(&data_bytes, bytes_used as usize)
            .map_err(ColError::from)?;

        // Update bytes_used in metadata
        let new_bytes_used = bytes_used + needed;
        conn.execute(
            "UPDATE col_chunks SET bytes_used = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
            params![new_bytes_used, col_id, chunk_id],
        )
        .map_err(ColError::from)?;

        Ok((chunk_id as i32, bytes_used as i32, new_bytes_used as i32))
    }

    /// Extend variable-size column with multiple elements (FAST - chunk-wise writes!)
    ///
    /// Strategy:
    /// 1. Read last chunk's unused data if applicable
    /// 2. Buffer elements until buffer reaches max_chunk_size
    /// 3. Write ENTIRE chunk at once (not element-by-element!)
    /// 4. Last buffer uses nearest legal_chunk_size
    ///
    /// Returns: Packed index data (12 bytes per element)
    fn extend_adaptive(
        &self,
        py: Python<'_>,
        data_col_id: i64,
        values: &Bound<'_, PyList>,
        max_chunk_bytes: i64,
    ) -> PyResult<Py<PyBytes>> {
        let conn = self.conn.lock().unwrap();

        // Get min_chunk_bytes
        let min_chunk_bytes: i64 = conn
            .query_row(
                "SELECT min_chunk_bytes FROM col_meta WHERE col_id = ?1",
                params![data_col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        let mut index_data = Vec::new();
        let mut buffer: Vec<Vec<u8>> = Vec::new();
        let mut buffer_size = 0i64;

        // Step 1: Check if last chunk has unused space
        let (mut current_chunk_id, last_chunk_unused) = match conn.query_row(
            "SELECT chunk_idx, actual_size, bytes_used FROM col_chunks
             WHERE col_id = ?1 ORDER BY chunk_idx DESC LIMIT 1",
            params![data_col_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        ) {
            Ok((chunk_id, chunk_size, bytes_used)) => {
                if bytes_used < chunk_size && chunk_size <= max_chunk_bytes {
                    // Read unused portion to combine with new data
                    let unused = (chunk_size - bytes_used) as usize;
                    (chunk_id, unused)
                } else {
                    (chunk_id, 0)
                }
            }
            Err(_) => (0, 0),
        };

        // If we have unused space, we'll overwrite the last chunk
        let _first_write_overwrites_last = last_chunk_unused > 0;

        // Step 2: Buffer elements and write full chunks
        for value in values.iter() {
            let elem_bytes = value.downcast::<PyBytes>()?.as_bytes().to_vec();
            let elem_len = elem_bytes.len() as i64;

            // Check if adding this element exceeds max_chunk_size
            if buffer_size + elem_len > max_chunk_bytes && !buffer.is_empty() {
                // Write buffered data as full chunk
                let chunk_data: Vec<u8> = buffer.concat();

                // Create chunk with legal size (capacity)
                let mut chunk_capacity = min_chunk_bytes;
                while chunk_capacity < buffer_size && chunk_capacity < max_chunk_bytes {
                    chunk_capacity *= 2;
                }
                chunk_capacity = std::cmp::min(chunk_capacity, max_chunk_bytes);

                current_chunk_id += 1;

                // Create chunk with zeroblob capacity
                conn.execute(
                    "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size, bytes_used)
                     VALUES (?1, ?2, zeroblob(?3), ?3, ?4)",
                    params![data_col_id, current_chunk_id, chunk_capacity, buffer_size],
                )
                .map_err(ColError::from)?;

                // Write data using BLOB API
                let rowid: i64 = conn
                    .query_row(
                        "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                        params![data_col_id, current_chunk_id],
                        |row| row.get(0),
                    )
                    .map_err(ColError::from)?;

                let mut blob = conn
                    .blob_open(rusqlite::DatabaseName::Main, "col_chunks", "data", rowid, false)
                    .map_err(ColError::from)?;

                blob.write_at(&chunk_data, 0).map_err(ColError::from)?;

                // Build index entries for this chunk
                let mut offset = 0i32;
                for elem in &buffer {
                    let start = offset;
                    let end = offset + elem.len() as i32;
                    let packed = Self::pack_index_triple(current_chunk_id as i32, start, end);
                    index_data.extend_from_slice(&packed);
                    offset = end;
                }

                // Clear buffer
                buffer.clear();
                buffer_size = 0;
            }

            // Add element to buffer
            buffer.push(elem_bytes);
            buffer_size += elem_len;
        }

        // Step 3: Write remaining buffer with legal chunk size
        if !buffer.is_empty() {
            let chunk_data: Vec<u8> = buffer.concat();

            // Find legal chunk size (capacity)
            let mut chunk_capacity = min_chunk_bytes;
            while chunk_capacity < buffer_size && chunk_capacity < max_chunk_bytes {
                chunk_capacity *= 2;
            }
            chunk_capacity = std::cmp::min(chunk_capacity, max_chunk_bytes);

            current_chunk_id += 1;

            // Create chunk with zeroblob capacity
            conn.execute(
                "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size, bytes_used)
                 VALUES (?1, ?2, zeroblob(?3), ?3, ?4)",
                params![data_col_id, current_chunk_id, chunk_capacity, buffer_size],
            )
            .map_err(ColError::from)?;

            // Write data using BLOB API
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![data_col_id, current_chunk_id],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            let mut blob = conn
                .blob_open(rusqlite::DatabaseName::Main, "col_chunks", "data", rowid, false)
                .map_err(ColError::from)?;

            blob.write_at(&chunk_data, 0).map_err(ColError::from)?;

            // Build index entries
            let mut offset = 0i32;
            for elem in &buffer {
                let start = offset;
                let end = offset + elem.len() as i32;
                let packed = Self::pack_index_triple(current_chunk_id as i32, start, end);
                index_data.extend_from_slice(&packed);
                offset = end;
            }
        }

        Ok(PyBytes::new_bound(py, &index_data).unbind())
    }

    /// Delete element from variable-size column (marks chunk as having deletions)
    ///
    /// Note: Doesn't actually remove data, just marks for vacuum
    fn delete_adaptive(&self, idx_col_id: i64, elem_idx: i64) -> PyResult<i32> {
        let conn = self.conn.lock().unwrap();

        // Read index entry to get chunk_id
        let index_data: Vec<u8> = conn
            .query_row(
                "SELECT data FROM col_chunks WHERE col_id = ?1",
                params![idx_col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Extract chunk_id from elem_idx position
        let offset = (elem_idx * 12) as usize;
        let chunk_id = i32::from_le_bytes([
            index_data[offset],
            index_data[offset + 1],
            index_data[offset + 2],
            index_data[offset + 3],
        ]);

        // Mark chunk as having deletions
        conn.execute(
            "UPDATE col_chunks SET has_deleted = 1 WHERE chunk_idx = ?1",
            params![chunk_id as i64],
        )
        .map_err(ColError::from)?;

        Ok(chunk_id)
    }

    /// Manually checkpoint WAL file to main database.
    /// This helps prevent WAL from growing too large.
    /// Returns success indicator (0 = success).
    fn checkpoint_wal(&self) -> PyResult<i64> {
        let conn = self.conn.lock().unwrap();
        // PRAGMA wal_checkpoint(PASSIVE) - non-blocking checkpoint
        conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")
            .map_err(ColError::from)?;
        Ok(0) // Success indicator
    }
}

// Helper methods (not exposed to Python)
impl _ColumnVault {
    /// Pack (i32, i32, i32) index triple to 12 bytes (little-endian)
    fn pack_index_triple(chunk_id: i32, start: i32, end: i32) -> [u8; 12] {
        let mut result = [0u8; 12];
        result[0..4].copy_from_slice(&chunk_id.to_le_bytes());
        result[4..8].copy_from_slice(&start.to_le_bytes());
        result[8..12].copy_from_slice(&end.to_le_bytes());
        result
    }

    /// Align chunk sizes to element size boundaries.
    /// Returns (aligned_min, aligned_max) or error if impossible.
    fn align_chunk_sizes(elem_size: i64, min: i64, max: i64) -> PyResult<(i64, i64)> {
        // Align min to next multiple of elem_size
        let aligned_min = ((min + elem_size - 1) / elem_size) * elem_size;

        // Align max to previous multiple of elem_size
        let aligned_max = (max / elem_size) * elem_size;

        // Check if alignment is valid
        if aligned_min > aligned_max {
            return Err(ColError::Col(format!(
                "Cannot align chunk sizes: elem_size={}, min={}, max={} -> aligned_min={} > aligned_max={}. \
                 Please increase max_chunk_bytes.",
                elem_size, min, max, aligned_min, aligned_max
            )).into());
        }

        Ok((aligned_min, aligned_max))
    }

    /// Prepare a chunk for appending data.
    /// Returns (chunk_idx, offset_in_chunk, chunk_capacity).
    ///
    /// Strategy:
    /// 1. Check if current chunk has space → return it
    /// 2. If not, try to DOUBLE current chunk (if < max)
    /// 3. If need new chunk: grow current to MAX, create new with nearest size
    fn prepare_append_chunk(
        &self,
        conn: &Connection,
        col_id: i64,
        current_byte_offset: i64,
        remaining_bytes: usize,
        min_chunk_bytes: i64,
        max_chunk_bytes: i64,
    ) -> PyResult<(i64, usize, usize)> {
        // Get last chunk if exists
        let last_chunk: Result<(i64, i64), _> = conn.query_row(
            "SELECT chunk_idx, actual_size FROM col_chunks WHERE col_id = ?1 ORDER BY chunk_idx DESC LIMIT 1",
            params![col_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        );

        match last_chunk {
            Ok((chunk_idx, actual_size)) => {
                // Calculate how much of this chunk is used
                let bytes_before_chunk = self.get_bytes_before_chunk(conn, col_id, chunk_idx)?;
                let bytes_in_chunk = current_byte_offset - bytes_before_chunk;
                let space_left = (actual_size - bytes_in_chunk) as usize;

                // Step 1: If current chunk has enough space, use it
                if space_left >= remaining_bytes {
                    return Ok((chunk_idx, bytes_in_chunk as usize, actual_size as usize));
                }

                // Step 2: Not enough space, try to grow current chunk
                if actual_size < max_chunk_bytes {
                    // Try to DOUBLE the size (not jump to max)
                    let new_size = std::cmp::min(actual_size * 2, max_chunk_bytes);
                    self.grow_chunk_to_size(conn, col_id, chunk_idx, new_size)?;
                    let new_space = (new_size - bytes_in_chunk) as usize;

                    // After doubling, check if enough space
                    if new_space >= remaining_bytes {
                        return Ok((chunk_idx, bytes_in_chunk as usize, new_size as usize));
                    }

                    // Still not enough, grow to MAX before creating new chunk
                    if new_size < max_chunk_bytes {
                        self.grow_chunk_to_size(conn, col_id, chunk_idx, max_chunk_bytes)?;
                    }
                    // After growing to max, return current chunk to fill it first
                    let space_at_max = (max_chunk_bytes - bytes_in_chunk) as usize;
                    if space_at_max > 0 {
                        // CRITICAL: Fill remaining space in current chunk before moving to next
                        return Ok((chunk_idx, bytes_in_chunk as usize, max_chunk_bytes as usize));
                    }
                    // Chunk is completely full, will create new chunk below
                } else if space_left > 0 {
                    // Chunk is at max but not completely full - fill it first!
                    return Ok((chunk_idx, bytes_in_chunk as usize, actual_size as usize));
                }

                // Step 3: Current chunk is at max and COMPLETELY full, create new chunk
                // All previous chunks (0..chunk_idx) are now at max_chunk_bytes
                let new_chunk_idx = chunk_idx + 1;
                let new_chunk_size = self.calculate_new_chunk_size(
                    remaining_bytes,
                    min_chunk_bytes,
                    max_chunk_bytes,
                );
                self.create_chunk_with_size(conn, col_id, new_chunk_idx, new_chunk_size)?;
                Ok((new_chunk_idx, 0, new_chunk_size as usize))
            }
            Err(_) => {
                // No chunks yet, create first one
                let chunk_size = self.calculate_new_chunk_size(
                    remaining_bytes,
                    min_chunk_bytes,
                    max_chunk_bytes,
                );
                self.create_chunk_with_size(conn, col_id, 0, chunk_size)?;
                Ok((0, 0, chunk_size as usize))
            }
        }
    }

    /// Calculate appropriate size for new chunk based on remaining data.
    /// Uses exponential growth: 2^k * min_chunk_bytes, capped at max_chunk_bytes.
    fn calculate_new_chunk_size(
        &self,
        remaining_bytes: usize,
        min_chunk: i64,
        max_chunk: i64,
    ) -> i64 {
        let remaining = remaining_bytes as i64;

        // If remaining fits in max, find nearest power of 2
        if remaining <= max_chunk {
            let mut size = min_chunk;
            while size < remaining && size < max_chunk {
                size *= 2;
            }
            std::cmp::min(size, max_chunk)
        } else {
            // Remaining is larger than max, use max
            max_chunk
        }
    }

    /// Create a chunk with specific size.
    fn create_chunk_with_size(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
        size: i64,
    ) -> PyResult<()> {
        let zeroblob = vec![0u8; size as usize];
        conn.execute(
            "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size) VALUES (?1, ?2, ?3, ?4)",
            params![col_id, chunk_idx, zeroblob, size],
        )
        .map_err(ColError::from)?;
        Ok(())
    }

    /// Grow a chunk to specified size (e.g., double, or to max).
    /// CRITICAL: Check actual_size first to avoid unnecessary blob reads.
    fn grow_chunk_to_size(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
        target_size: i64,
    ) -> PyResult<()> {
        // CRITICAL: Check actual_size FIRST to avoid reading large blobs unnecessarily
        let actual_size: i64 = conn
            .query_row(
                "SELECT actual_size FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Already at or above target size, no need to grow
        if actual_size >= target_size {
            return Ok(());
        }

        // Read blob data for growing
        let old_data: Vec<u8> = conn
            .query_row(
                "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        let mut new_data = vec![0u8; target_size as usize];
        new_data[..old_data.len()].copy_from_slice(&old_data);

        conn.execute(
            "UPDATE col_chunks SET data = ?1, actual_size = ?2 WHERE col_id = ?3 AND chunk_idx = ?4",
            params![new_data, target_size, col_id, chunk_idx],
        )
        .map_err(ColError::from)?;

        Ok(())
    }

    /// Get or create a chunk.
    /// For append operations (use_min_size=true): start with min_chunk_bytes
    /// For random access (use_min_size=false): create at max_chunk_bytes for immediate use
    fn ensure_chunk(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
        use_min_size: bool,
    ) -> PyResult<()> {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            // Get chunk size settings for this column
            let (min_chunk, max_chunk): (i64, i64) = conn
                .query_row(
                    "SELECT min_chunk_bytes, max_chunk_bytes FROM col_meta WHERE col_id = ?1",
                    params![col_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(ColError::from)?;

            // Create new chunk at min or max size depending on use case
            let chunk_size = if use_min_size { min_chunk } else { max_chunk };
            let zeroblob = vec![0u8; chunk_size as usize];
            conn.execute(
                "
                INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size)
                VALUES (?1, ?2, ?3, ?4)
                ",
                params![col_id, chunk_idx, zeroblob, chunk_size],
            )
            .map_err(ColError::from)?;
        }

        Ok(())
    }

    /// Calculate total bytes stored before a given chunk.
    fn get_bytes_before_chunk(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
    ) -> PyResult<i64> {
        if chunk_idx == 0 {
            return Ok(0);
        }

        // Sum up actual_size of all previous chunks
        let total: i64 = conn
            .query_row(
                "
                SELECT COALESCE(SUM(actual_size), 0)
                FROM col_chunks
                WHERE col_id = ?1 AND chunk_idx < ?2
                ",
                params![col_id, chunk_idx],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        Ok(total)
    }
}
