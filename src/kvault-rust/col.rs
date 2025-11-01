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
    ///     (col_id, elem_size, length, chunk_bytes)
    fn get_column_info(&self, name: &str) -> PyResult<(i64, i64, i64, i64)> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "
            SELECT col_id, elem_size, length, chunk_bytes
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

        let elems_per_chunk = chunk_bytes / elem_size;
        let start_chunk = start_idx / elems_per_chunk;
        let end_idx = start_idx + count;
        let end_chunk = (end_idx - 1) / elems_per_chunk;

        let total_bytes = (count * elem_size) as usize;
        let mut result = vec![0u8; total_bytes];
        let mut bytes_written = 0usize;

        for chunk_idx in start_chunk..=end_chunk {
            // Calculate which elements from this chunk we need
            let chunk_start_elem = chunk_idx * elems_per_chunk;
            let chunk_end_elem = chunk_start_elem + elems_per_chunk;

            let read_start = std::cmp::max(start_idx, chunk_start_elem);
            let read_end = std::cmp::min(end_idx, chunk_end_elem);
            let read_count = read_end - read_start;

            if read_count <= 0 {
                continue;
            }

            // Offset within chunk
            let offset_in_chunk = ((read_start - chunk_start_elem) * elem_size) as usize;
            let bytes_to_read = (read_count * elem_size) as usize;

            // Read from chunk
            let chunk_data: Vec<u8> = conn
                .query_row(
                    "
                    SELECT data
                    FROM col_chunks
                    WHERE col_id = ?1 AND chunk_idx = ?2
                    ",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Copy relevant portion
            result[bytes_written..bytes_written + bytes_to_read]
                .copy_from_slice(&chunk_data[offset_in_chunk..offset_in_chunk + bytes_to_read]);

            bytes_written += bytes_to_read;
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
        let count = (data_bytes.len() as i64) / elem_size;

        let elems_per_chunk = chunk_bytes / elem_size;
        let start_chunk = start_idx / elems_per_chunk;
        let end_idx = start_idx + count;
        let end_chunk = (end_idx - 1) / elems_per_chunk;

        let mut bytes_read = 0usize;

        for chunk_idx in start_chunk..=end_chunk {
            let chunk_start_elem = chunk_idx * elems_per_chunk;
            let chunk_end_elem = chunk_start_elem + elems_per_chunk;

            let write_start = std::cmp::max(start_idx, chunk_start_elem);
            let write_end = std::cmp::min(end_idx, chunk_end_elem);
            let write_count = write_end - write_start;

            if write_count <= 0 {
                continue;
            }

            let offset_in_chunk = ((write_start - chunk_start_elem) * elem_size) as usize;
            let bytes_to_write = (write_count * elem_size) as usize;

            // Ensure chunk exists
            self.ensure_chunk(&conn, col_id, chunk_idx, chunk_bytes as usize)?;

            // Update chunk data
            let mut chunk_data: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            chunk_data[offset_in_chunk..offset_in_chunk + bytes_to_write]
                .copy_from_slice(&data_bytes[bytes_read..bytes_read + bytes_to_write]);

            conn.execute(
                "UPDATE col_chunks SET data = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                params![chunk_data, col_id, chunk_idx],
            )
            .map_err(ColError::from)?;

            bytes_read += bytes_to_write;
        }

        Ok(())
    }

    /// Append raw bytes to the end of a column with dynamic chunk growth.
    /// Most performance-critical operation.
    ///
    /// Strategy:
    /// - Append to last chunk if there's space
    /// - If full, grow chunk 2x (up to max_chunk_bytes)
    /// - If already at max, create new chunk at min_chunk_bytes
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
        let bytes_to_append = data_bytes.len();

        // For raw bytes (elem_size=1), current_length is in bytes
        // For typed data, current_length is in elements
        let current_byte_offset = if elem_size == 1 {
            current_length
        } else {
            current_length * elem_size
        };

        // Find or create chunk with space
        let (target_chunk_idx, offset_in_chunk) =
            self.find_append_chunk(&conn, col_id, current_byte_offset, bytes_to_append)?;

        // Read chunk data
        let mut chunk_data: Vec<u8> = conn
            .query_row(
                "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, target_chunk_idx],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Write data to chunk
        let end_offset = offset_in_chunk + bytes_to_append;
        chunk_data[offset_in_chunk..end_offset].copy_from_slice(data_bytes);

        conn.execute(
            "UPDATE col_chunks SET data = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
            params![chunk_data, col_id, target_chunk_idx],
        )
        .map_err(ColError::from)?;

        // Update length
        let new_length = if elem_size == 1 {
            current_length + bytes_to_append as i64
        } else {
            current_length + (bytes_to_append as i64 / elem_size)
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
    /// Get or create a chunk, starting with min size.
    /// New chunks always start at min_chunk_bytes.
    fn ensure_chunk(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
        _default_chunk_bytes: usize,
    ) -> PyResult<()> {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            // Get min chunk size for this column
            let min_chunk: i64 = conn
                .query_row(
                    "SELECT min_chunk_bytes FROM col_meta WHERE col_id = ?1",
                    params![col_id],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Create new chunk at min size
            let zeroblob = vec![0u8; min_chunk as usize];
            conn.execute(
                "
                INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size)
                VALUES (?1, ?2, ?3, ?4)
                ",
                params![col_id, chunk_idx, zeroblob, min_chunk],
            )
            .map_err(ColError::from)?;
        }

        Ok(())
    }

    /// Grow a chunk in-place by doubling its size (up to max).
    /// Returns true if grown, false if already at max.
    fn try_grow_chunk(&self, conn: &Connection, col_id: i64, chunk_idx: i64) -> PyResult<bool> {
        // Get current size and max size
        let (current_size, max_chunk): (i64, i64) = conn
            .query_row(
                "
                SELECT c.actual_size, m.max_chunk_bytes
                FROM col_chunks c
                JOIN col_meta m ON c.col_id = m.col_id
                WHERE c.col_id = ?1 AND c.chunk_idx = ?2
                ",
                params![col_id, chunk_idx],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ColError::from)?;

        if current_size >= max_chunk {
            // Already at max size, can't grow
            return Ok(false);
        }

        // Double the size, capped at max
        let new_size = std::cmp::min(current_size * 2, max_chunk);

        // Read existing data
        let old_data: Vec<u8> = conn
            .query_row(
                "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Create new blob with doubled size (preserving old data)
        let mut new_data = vec![0u8; new_size as usize];
        new_data[..old_data.len()].copy_from_slice(&old_data);

        // Update chunk with new blob
        conn.execute(
            "
            UPDATE col_chunks
            SET data = ?1, actual_size = ?2
            WHERE col_id = ?3 AND chunk_idx = ?4
            ",
            params![new_data, new_size, col_id, chunk_idx],
        )
        .map_err(ColError::from)?;

        Ok(true)
    }

    /// Find which chunk to append to, growing or creating as needed.
    /// Returns (chunk_idx, offset_in_chunk).
    fn find_append_chunk(
        &self,
        conn: &Connection,
        col_id: i64,
        current_byte_offset: i64,
        bytes_needed: usize,
    ) -> PyResult<(i64, usize)> {
        // Get last chunk if exists
        let last_chunk: Result<(i64, i64), _> = conn.query_row(
            "
            SELECT chunk_idx, actual_size
            FROM col_chunks
            WHERE col_id = ?1
            ORDER BY chunk_idx DESC
            LIMIT 1
            ",
            params![col_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        );

        match last_chunk {
            Ok((chunk_idx, actual_size)) => {
                // Calculate how much of this chunk is used
                let bytes_before_chunk = self.get_bytes_before_chunk(conn, col_id, chunk_idx)?;
                let bytes_in_chunk = current_byte_offset - bytes_before_chunk;
                let space_left = (actual_size - bytes_in_chunk) as usize;

                if space_left >= bytes_needed {
                    // Enough space in current chunk
                    return Ok((chunk_idx, bytes_in_chunk as usize));
                }

                // Try to grow current chunk
                if self.try_grow_chunk(conn, col_id, chunk_idx)? {
                    // Successfully grew, try again
                    return self.find_append_chunk(conn, col_id, current_byte_offset, bytes_needed);
                }

                // Can't grow, create new chunk
                let new_chunk_idx = chunk_idx + 1;
                self.ensure_chunk(conn, col_id, new_chunk_idx, 0)?;
                Ok((new_chunk_idx, 0))
            }
            Err(_) => {
                // No chunks yet, create first one
                self.ensure_chunk(conn, col_id, 0, 0)?;
                Ok((0, 0))
            }
        }
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
