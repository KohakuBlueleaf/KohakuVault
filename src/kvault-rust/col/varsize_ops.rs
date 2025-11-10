// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Variable-size column operations.
//!
//! Handles reading, writing, and extending variable-size data columns.
//! Uses adaptive chunking strategy with bytes_used tracking.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyList};
use rusqlite::params;

use super::chunks::pack_index_triple;
use super::ColError;

impl super::_ColumnVault {
    /// Read from adaptive variable-size storage (single chunk, known offsets).
    pub(crate) fn read_adaptive_impl(
        &self,
        py: Python<'_>,
        col_id: i64,
        chunk_id: i32,
        start_byte: i32,
        end_byte: i32,
    ) -> PyResult<Py<PyBytes>> {
        // Release GIL during blocking I/O operations
        let result = py.allow_threads(|| {
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
            let mut result_vec = vec![0u8; len];
            blob.read_at(&mut result_vec, start_byte as usize)
                .map_err(ColError::from)?;

            Ok::<Vec<u8>, PyErr>(result_vec)
        })?;

        Ok(PyBytes::new_bound(py, &result).unbind())
    }

    /// Batch read variable-size elements with single FFI call.
    ///
    /// This function performs all the "find chunks + extract data" logic in Rust:
    /// 1. Read N index entries (N×12 bytes) from index column
    /// 2. Parse indices in Rust (fast bit manipulation)
    /// 3. Group reads by chunk_id (minimize SQLite queries)
    /// 4. For each unique chunk: read once, extract multiple elements
    /// 5. Return all element data in optimal format
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `idx_col_id` - Index column ID
    /// * `data_col_id` - Data column ID
    /// * `start_idx` - Starting element index
    /// * `count` - Number of elements to read
    /// * `idx_elem_size` - Should be 12 (for validation)
    /// * `idx_chunk_bytes` - Aligned chunk size for index column
    ///
    /// # Returns
    /// List of PyBytes (one per element, preserving order)
    ///
    /// # Performance
    /// Expected: 10-50x faster than Python loop for N=10-1000
    /// - Single index read + minimal data chunk reads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_read_varsize_impl(
        &self,
        py: Python<'_>,
        idx_col_id: i64,
        data_col_id: i64,
        start_idx: i64,
        count: i64,
        idx_elem_size: i64,
        idx_chunk_bytes: i64,
    ) -> PyResult<Py<PyList>> {
        // Validate inputs
        if idx_elem_size != 12 {
            return Err(ColError::Col(format!(
                "Invalid index elem_size: expected 12, got {}",
                idx_elem_size
            ))
            .into());
        }

        if count == 0 {
            return Ok(PyList::empty_bound(py).unbind());
        }

        if count > 10_000_000 {
            return Err(ColError::Col(format!(
                "Batch read too large: {} elements (max 10M)",
                count
            ))
            .into());
        }

        // Release GIL during all SQLite operations
        let index_data_bytes = py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();

            // STEP 1: Read all index entries at once (N×12 bytes)
            // This is ONE SQLite read, much faster than N individual reads
            self._read_range_internal(
                &conn,
                idx_col_id,
                start_idx,
                count,
                idx_elem_size,
                idx_chunk_bytes,
            )
        })?;

        // STEP 2: Parse all indices in Rust (fast!)
        // Build a map: chunk_id -> Vec<(start, end, result_idx)>
        let mut chunk_reads: HashMap<i32, Vec<(i32, i32, usize)>> = HashMap::new();

        for i in 0..count as usize {
            let offset = i * 12;
            let chunk_id = i32::from_le_bytes([
                index_data_bytes[offset],
                index_data_bytes[offset + 1],
                index_data_bytes[offset + 2],
                index_data_bytes[offset + 3],
            ]);
            let start_byte = i32::from_le_bytes([
                index_data_bytes[offset + 4],
                index_data_bytes[offset + 5],
                index_data_bytes[offset + 6],
                index_data_bytes[offset + 7],
            ]);
            let end_byte = i32::from_le_bytes([
                index_data_bytes[offset + 8],
                index_data_bytes[offset + 9],
                index_data_bytes[offset + 10],
                index_data_bytes[offset + 11],
            ]);

            chunk_reads
                .entry(chunk_id)
                .or_default()
                .push((start_byte, end_byte, i));
        }

        // STEP 3: Prepare result array (preserve order!)
        let mut results: Vec<Option<Vec<u8>>> = vec![None; count as usize];

        // STEP 4: For each unique chunk, read ONCE and extract all elements
        // Release GIL during blob I/O operations
        py.allow_threads(|| -> PyResult<()> {
            let conn = self.conn.lock().unwrap();

            for (chunk_id, reads) in chunk_reads.iter() {
                // Get rowid for BLOB read
                let rowid: i64 = conn
                    .query_row(
                        "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                        params![data_col_id, *chunk_id as i64],
                        |row| row.get(0),
                    )
                    .map_err(|e| {
                        ColError::Col(format!(
                            "Chunk not found: col_id={}, chunk_id={}, error={}",
                            data_col_id, chunk_id, e
                        ))
                    })?;

                // Open BLOB for efficient reading
                let blob = conn
                    .blob_open(
                        rusqlite::DatabaseName::Main,
                        "col_chunks",
                        "data",
                        rowid,
                        true, // read-only
                    )
                    .map_err(ColError::from)?;

                // Extract all elements from this chunk
                for &(start_byte, end_byte, result_idx) in reads {
                    let len = (end_byte - start_byte) as usize;
                    let mut element_data = vec![0u8; len];
                    blob.read_at(&mut element_data, start_byte as usize)
                        .map_err(ColError::from)?;

                    results[result_idx] = Some(element_data);
                }
            }

            Ok(())
        })?;

        // STEP 5: Convert to Python list
        let py_list = PyList::empty_bound(py);
        for result in results {
            match result {
                Some(data) => {
                    py_list.append(PyBytes::new_bound(py, &data))?;
                }
                None => {
                    return Err(ColError::Col("Missing element in batch read".to_string()).into());
                }
            }
        }

        Ok(py_list.unbind())
    }

    /// Batch read + unpack variable-size elements in one call.
    ///
    /// Combines batch reading with DataPacker unpacking for maximum efficiency.
    ///
    /// # Arguments
    /// * `packer` - DataPacker instance for unpacking
    /// * Other args same as batch_read_varsize
    ///
    /// # Returns
    /// List of unpacked Python objects (not bytes)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_read_varsize_unpacked_impl(
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
        // First, get raw bytes using batch_read_varsize_impl logic
        let raw_list = self.batch_read_varsize_impl(
            py,
            idx_col_id,
            data_col_id,
            start_idx,
            count,
            idx_elem_size,
            idx_chunk_bytes,
        )?;

        // Unpack each element using DataPacker (all in Rust!)
        let raw_list_bound = raw_list.bind(py);
        let result_list = PyList::empty_bound(py);

        for item in raw_list_bound.iter() {
            let bytes = item.downcast::<PyBytes>()?;
            let unpacked = packer.unpack(py, bytes.as_bytes(), 0)?;
            result_list.append(unpacked)?;
        }

        Ok(result_list.unbind())
    }

    /// Append raw data to variable-size column with adaptive chunking.
    /// Returns (chunk_id, start_byte, end_byte) as i32 triple.
    ///
    /// New strategy (v0.4.0):
    /// - Tracks bytes_used vs chunk_size (capacity)
    /// - Only expands when truly needed
    /// - Smart expansion based on element size
    pub(crate) fn append_raw_adaptive_impl(
        &self,
        py: Python<'_>,
        col_id: i64,
        data: &Bound<'_, PyBytes>,
        max_chunk_bytes: i64,
    ) -> PyResult<(i32, i32, i32)> {
        let data_bytes = data.as_bytes().to_vec();
        let needed = data_bytes.len() as i64;

        // Release GIL during SQLite write operations
        py.allow_threads(|| {
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
        })
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
    pub(crate) fn extend_adaptive_impl(
        &self,
        py: Python<'_>,
        data_col_id: i64,
        values: &Bound<'_, PyList>,
        max_chunk_bytes: i64,
    ) -> PyResult<Py<PyBytes>> {
        // Convert PyList to Vec outside allow_threads
        let value_bytes: Vec<Vec<u8>> = values
            .iter()
            .map(|v| v.downcast::<PyBytes>().unwrap().as_bytes().to_vec())
            .collect();

        // Release GIL during all SQLite operations
        let index_data = py.allow_threads(|| {
            let conn = self.conn.lock().unwrap();

            // Get min_chunk_bytes
            let min_chunk_bytes: i64 = conn
                .query_row(
                    "SELECT min_chunk_bytes FROM col_meta WHERE col_id = ?1",
                    params![data_col_id],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

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

            let mut index_data: Vec<u8> = Vec::new();

            // Step 2: Buffer elements and write full chunks
            for elem_bytes in &value_bytes {
                let elem_len = elem_bytes.len() as i64;

                // Check if adding this element exceeds max_chunk_size
                if buffer_size + elem_len > max_chunk_bytes && !buffer.is_empty() {
                    // Write buffered data as full chunk
                    let chunk_data: Vec<u8> = buffer.concat();

                    // Create chunk with legal size (capacity) - find smallest power of 2
                    let mut chunk_capacity = min_chunk_bytes;
                    while chunk_capacity < buffer_size {
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
                        let packed = pack_index_triple(current_chunk_id as i32, start, end);
                        index_data.extend_from_slice(&packed);
                        offset = end;
                    }

                    // Clear buffer
                    buffer.clear();
                    buffer_size = 0;
                }

                // Add element to buffer
                buffer.push(elem_bytes.clone());
                buffer_size += elem_len;
            }

            // Step 3: Write remaining buffer with legal chunk size
            if !buffer.is_empty() {
                let chunk_data: Vec<u8> = buffer.concat();

                // Find legal chunk size (capacity) - find smallest power of 2
                let mut chunk_capacity = min_chunk_bytes;
                while chunk_capacity < buffer_size {
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
                    let packed = pack_index_triple(current_chunk_id as i32, start, end);
                    index_data.extend_from_slice(&packed);
                    offset = end;
                }
            }

            Ok::<Vec<u8>, PyErr>(index_data)
        })?;

        Ok(PyBytes::new_bound(py, &index_data).unbind())
    }

    /// Append raw data to cache for variable-size column (cached adaptive interface).
    /// This is used for variable-size columns to cache individual elements.
    /// Returns true if cache was auto-flushed.
    pub(crate) fn append_raw_adaptive_cached_impl(
        &self,
        data_col_id: i64,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<bool> {
        // Simply append to cache - flush will handle calling extend_adaptive_impl
        self.append_cached(data_col_id, data, 1, 0)
    }

    /// Extend cache for variable-size column with multiple elements (cached adaptive interface).
    /// Elements should be a list of PyBytes (packed elements).
    /// Returns true if cache was auto-flushed.
    pub(crate) fn extend_adaptive_cached_impl(
        &self,
        data_col_id: i64,
        elements: &Bound<'_, PyList>,
    ) -> PyResult<bool> {
        // Use extend_cached with is_variable_size=true
        self.extend_cached(data_col_id, elements, true)
    }
}
