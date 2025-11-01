use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};
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
        let _ = conn.pragma_update(None, "journal_mode", &"WAL");
        let _ = conn.pragma_update(None, "synchronous", &"NORMAL");

        // Create schema for columnar storage
        conn.execute_batch(
            "
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
                PRIMARY KEY (col_id, chunk_idx),
                FOREIGN KEY (col_id) REFERENCES col_meta(col_id) ON DELETE CASCADE
            );
            ",
        )
        .map_err(ColError::from)?;

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
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "
            INSERT INTO col_meta (name, dtype, elem_size, length, chunk_bytes, min_chunk_bytes, max_chunk_bytes)
            VALUES (?1, ?2, ?3, 0, ?4, ?5, ?6)
            ",
            params![name, dtype, elem_size, chunk_bytes, min_chunk_bytes, max_chunk_bytes],
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

        // CRITICAL: Use BYTE offsets for chunk addressing, not element counts!
        let start_byte = start_idx * elem_size;
        let total_bytes = (count * elem_size) as usize;
        let end_byte = start_byte + total_bytes as i64;

        // Calculate which chunks contain our byte range
        let start_chunk = start_byte / chunk_bytes;
        let end_chunk = (end_byte - 1) / chunk_bytes;

        let mut result = vec![0u8; total_bytes];
        let mut result_offset = 0usize;

        for chunk_idx in start_chunk..=end_chunk {
            // Calculate byte range for this chunk
            let chunk_start_byte = chunk_idx * chunk_bytes;
            let chunk_end_byte = chunk_start_byte + chunk_bytes;

            // Calculate overlap between our range and this chunk
            let read_start_byte = std::cmp::max(start_byte, chunk_start_byte);
            let read_end_byte = std::cmp::min(end_byte, chunk_end_byte);
            let bytes_to_read = (read_end_byte - read_start_byte) as usize;

            if bytes_to_read == 0 {
                continue;
            }

            // Offset within chunk (in bytes)
            let offset_in_chunk = (read_start_byte - chunk_start_byte) as usize;

            // Read chunk data
            let chunk_data: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Copy the portion we need
            result[result_offset..result_offset + bytes_to_read]
                .copy_from_slice(&chunk_data[offset_in_chunk..offset_in_chunk + bytes_to_read]);

            result_offset += bytes_to_read;
        }

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

        // CRITICAL: Use BYTE offsets for chunk addressing, not element counts!
        let start_byte = start_idx * elem_size;
        let total_bytes = data_bytes.len();
        let end_byte = start_byte + total_bytes as i64;

        // Calculate which chunks contain our byte range
        let start_chunk = start_byte / chunk_bytes;
        let end_chunk = (end_byte - 1) / chunk_bytes;

        let mut data_offset = 0usize;

        for chunk_idx in start_chunk..=end_chunk {
            // Calculate byte range for this chunk
            let chunk_start_byte = chunk_idx * chunk_bytes;
            let chunk_end_byte = chunk_start_byte + chunk_bytes;

            // Calculate overlap between our range and this chunk
            let write_start_byte = std::cmp::max(start_byte, chunk_start_byte);
            let write_end_byte = std::cmp::min(end_byte, chunk_end_byte);
            let bytes_to_write = (write_end_byte - write_start_byte) as usize;

            if bytes_to_write == 0 {
                continue;
            }

            // Offset within chunk (in bytes)
            let offset_in_chunk = (write_start_byte - chunk_start_byte) as usize;

            // Ensure chunk exists (create at max size for random access)
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
}

// Helper methods (not exposed to Python)
impl _ColumnVault {
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
