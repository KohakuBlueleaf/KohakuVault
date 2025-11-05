// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Fixed-size column operations.
//!
//! Handles reading and writing fixed-size elements across chunks.
//! All functions work with element-aligned chunks where elements don't cross boundaries.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyList};
use rusqlite::{params, Connection};

use super::ColError;

impl super::_ColumnVault {
    /// Read a range of elements from a fixed-size column.
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
    pub(crate) fn read_range_impl(
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

    /// Write a range of elements to a column (from raw bytes).
    ///
    /// Args:
    ///     col_id: Column ID
    ///     start_idx: Starting element index
    ///     data: Raw bytes to write
    ///     elem_size: Size of each element in bytes
    ///     chunk_bytes: Chunk size in bytes
    pub(crate) fn write_range_impl(
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

            // Ensure chunk exists and has enough capacity
            // For setitem operations, we might need to grow the chunk
            let chunk_exists: bool = conn
                .query_row(
                    "SELECT 1 FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |_| Ok(true),
                )
                .unwrap_or(false);

            if !chunk_exists {
                // Create chunk at appropriate size
                let (min_chunk, _max_chunk): (i64, i64) = conn
                    .query_row(
                        "SELECT min_chunk_bytes, max_chunk_bytes FROM col_meta WHERE col_id = ?1",
                        params![col_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .map_err(ColError::from)?;

                // For random writes, start with min and grow as needed
                let initial_size = min_chunk;
                // CRITICAL FIX: Use SQLite's zeroblob() to avoid memory allocation
                conn.execute(
                    "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size)
                     VALUES (?1, ?2, zeroblob(?3), ?3)",
                    params![col_id, chunk_idx, initial_size],
                )
                .map_err(ColError::from)?;
            }

            // Check if chunk needs to grow to accommodate this write
            let actual_size: i64 = conn
                .query_row(
                    "SELECT actual_size FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            let needed_size = offset_in_chunk + bytes_to_write;

            if needed_size > actual_size as usize {
                // Need to grow chunk
                let (_min_chunk, max_chunk): (i64, i64) = conn
                    .query_row(
                        "SELECT min_chunk_bytes, max_chunk_bytes FROM col_meta WHERE col_id = ?1",
                        params![col_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .map_err(ColError::from)?;

                // Grow to next power of 2 or max
                let mut new_size = actual_size;
                while new_size < needed_size as i64 && new_size < max_chunk {
                    new_size *= 2;
                }
                new_size = std::cmp::min(new_size, max_chunk);

                if needed_size > new_size as usize {
                    new_size = needed_size as i64; // Use exact size if still not enough
                }

                self.grow_chunk_to_size(&conn, col_id, chunk_idx, new_size)?;
            }

            // Use BLOB API for efficient write (doesn't read entire chunk!)
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![col_id, chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            let mut blob = conn
                .blob_open(
                    rusqlite::DatabaseName::Main,
                    "col_chunks",
                    "data",
                    rowid,
                    false, // writable
                )
                .map_err(ColError::from)?;

            blob.write_at(&data_bytes[data_offset..data_offset + bytes_to_write], offset_in_chunk)
                .map_err(ColError::from)?;

            data_offset += bytes_to_write;
        }

        Ok(())
    }

    /// Batch read fixed-size elements with optional unpacking.
    ///
    /// While read_range already handles this efficiently, this adds
    /// integrated unpacking for consistency with varsize API.
    ///
    /// # Arguments
    /// * `col_id` - Column ID
    /// * `start_idx` - Starting element index
    /// * `count` - Number of elements
    /// * `elem_size` - Size of each element
    /// * `chunk_bytes` - Chunk size
    /// * `packer` - Optional DataPacker for unpacking
    ///
    /// # Returns
    /// List of unpacked values
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_read_fixed_impl(
        &self,
        py: Python<'_>,
        col_id: i64,
        start_idx: i64,
        count: i64,
        elem_size: i64,
        chunk_bytes: i64,
        packer: Option<&crate::packer::DataPacker>,
    ) -> PyResult<Py<PyList>> {
        if count == 0 {
            return Ok(PyList::empty_bound(py).unbind());
        }

        // Use existing read_range_impl
        let data = self.read_range_impl(py, col_id, start_idx, count, elem_size, chunk_bytes)?;

        // If no packer, manually split and return bytes
        if packer.is_none() {
            let data_bytes = data.bind(py).as_bytes();
            let result = PyList::empty_bound(py);
            for i in 0..count as usize {
                let offset = i * elem_size as usize;
                let elem_bytes = &data_bytes[offset..offset + elem_size as usize];
                result.append(PyBytes::new_bound(py, elem_bytes))?;
            }
            return Ok(result.unbind());
        }

        // Unpack using DataPacker
        let packer = packer.unwrap();
        packer.unpack_many(py, data.bind(py).as_bytes(), Some(count as usize), None)
    }

    /// Append raw bytes to a fixed-size column.
    ///
    /// Handles chunk management and incremental writes using BLOB API.
    ///
    /// # Arguments
    /// * `col_id` - Column ID
    /// * `data` - Raw bytes to append
    /// * `elem_size` - Size of each element (1 for raw bytes)
    /// * `_chunk_bytes` - Chunk size (not used, kept for API compatibility)
    /// * `current_length` - Current column length
    pub(crate) fn append_raw_impl(
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
            // CRITICAL: Last parameter must be FALSE for write access!
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
    pub(crate) fn set_length_impl(&self, col_id: i64, new_length: i64) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE col_meta SET length = ?1 WHERE col_id = ?2",
            params![new_length, col_id],
        )
        .map_err(ColError::from)?;

        Ok(())
    }

    /// Append value using DataPacker (typed interface).
    pub(crate) fn append_typed_impl(
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
        self.append_raw_impl(col_id, py_bytes, elem_size, chunk_bytes, current_length)
    }

    /// Extend with multiple values using DataPacker (typed interface).
    pub(crate) fn extend_typed_impl(
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
        self.append_raw_impl(col_id, py_bytes, elem_size, chunk_bytes, current_length)
    }

    /// Append value to cache using DataPacker (cached typed interface).
    pub(crate) fn append_typed_cached_impl(
        &self,
        col_id: i64,
        value: &Bound<'_, PyAny>,
        packer: &crate::packer::DataPacker,
        current_length: i64,
    ) -> PyResult<bool> {
        // Pack value in Rust
        let packed_bytes = packer.pack(value.py(), value)?;
        let py_bytes = packed_bytes.bind(value.py());

        // Get element size from packer
        let elem_size = packer.elem_size() as i64;

        // Append to cache
        self.append_cached(col_id, py_bytes, elem_size, current_length)
    }

    /// Extend cache with multiple values using DataPacker (cached typed interface).
    pub(crate) fn extend_typed_cached_impl(
        &self,
        col_id: i64,
        values: &Bound<'_, PyList>,
        packer: &crate::packer::DataPacker,
    ) -> PyResult<bool> {
        // Pack all values in Rust
        let packed_bytes = packer.pack_many(values.py(), values)?;

        // Convert to list of individual packed elements
        let elem_size = packer.elem_size();
        let all_bytes = packed_bytes.bind(values.py()).as_bytes();

        // Split into individual elements
        let mut elements = Vec::new();
        for i in 0..(all_bytes.len() / elem_size) {
            let start = i * elem_size;
            let end = start + elem_size;
            elements.push(all_bytes[start..end].to_vec());
        }

        // Create PyList of PyBytes for extend_cached
        let py_list = PyList::new_bound(
            values.py(),
            elements
                .iter()
                .map(|e| PyBytes::new_bound(values.py(), e))
                .collect::<Vec<_>>(),
        );

        // Extend cache (not variable-size since these are fixed-size packed elements)
        self.extend_cached(col_id, &py_list, false)
    }

    /// Internal helper for reading range data (no Python overhead).
    /// Used by batch operations that need raw bytes.
    pub(crate) fn _read_range_internal(
        &self,
        conn: &Connection,
        col_id: i64,
        start_idx: i64,
        count: i64,
        elem_size: i64,
        chunk_bytes: i64,
    ) -> Result<Vec<u8>, ColError> {
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

        // Read from each chunk
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

        Ok(result)
    }
}
