// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Variable-size index management.
//!
//! Handles updating, deleting, and managing index entries for variable-size columns.
//! Index entries are 12-byte triples: (chunk_id, start_byte, end_byte)

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyList};
use rusqlite::{params, Connection};

use super::chunks::pack_index_triple;
use super::ColError;

impl super::_ColumnVault {
    /// Update variable-size element with size-aware logic (v0.5.0).
    ///
    /// Strategy:
    /// - new_size ≤ old_size: Direct replace, update index, DON'T change bytes_used (leave fragment!)
    /// - new_size > old_size: Treat as "insert after delete", update bytes_used, may rebuild/split chunk
    ///
    /// # Arguments
    /// * `data_col_id` - Data column ID
    /// * `idx_col_id` - Index column ID
    /// * `elem_idx` - Element index to update
    /// * `new_data` - New element data
    /// * `chunk_id` - Current chunk ID (from index)
    /// * `old_start` - Current start byte (from index)
    /// * `old_end` - Current end byte (from index)
    /// * `max_chunk_bytes` - Max chunk size
    ///
    /// # Returns
    /// (new_chunk_id, new_start, new_end)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_varsize_element_impl(
        &self,
        _py: Python<'_>,
        data_col_id: i64,
        idx_col_id: i64,
        elem_idx: i64,
        new_data: &Bound<'_, PyBytes>,
        chunk_id: i32,
        old_start: i32,
        old_end: i32,
        max_chunk_bytes: i64,
    ) -> PyResult<(i32, i32, i32)> {
        let conn = self.conn.lock().unwrap();
        let new_data_bytes = new_data.as_bytes();
        let new_size = new_data_bytes.len() as i32;
        let old_size = old_end - old_start;
        let size_delta = new_size - old_size;

        // Get chunk info
        let (actual_size, bytes_used): (i64, i64) = conn
            .query_row(
                "SELECT actual_size, bytes_used FROM col_chunks
                 WHERE col_id = ?1 AND chunk_idx = ?2",
                params![data_col_id, chunk_id as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ColError::from)?;

        // CASE 1: Same size or smaller - Direct replace
        if size_delta <= 0 {
            // Get rowid for BLOB API
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![data_col_id, chunk_id as i64],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Write new data using BLOB API
            let mut blob = conn
                .blob_open(
                    rusqlite::DatabaseName::Main,
                    "col_chunks",
                    "data",
                    rowid,
                    false, // writable
                )
                .map_err(ColError::from)?;

            blob.write_at(new_data_bytes, old_start as usize)
                .map_err(ColError::from)?;

            drop(blob);

            // Update index entry (new end_byte)
            let new_index = pack_index_triple(chunk_id, old_start, old_start + new_size);
            Self::update_index_entry_internal(
                &conn,
                idx_col_id,
                elem_idx,
                &new_index,
                max_chunk_bytes,
            )?;

            // IMPORTANT: DON'T update bytes_used (leave fragment for vacuum!)

            return Ok((chunk_id, old_start, old_start + new_size));
        }

        // CASE 2: Larger - treat as "delete old + insert new"
        // This requires rebuilding the chunk section after this element

        // Check if delta fits in remaining space
        let available = actual_size - bytes_used;

        if size_delta as i64 <= available {
            // Fits - shift elements after this one
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![data_col_id, chunk_id as i64],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Read data after old element
            let data_after_len = (bytes_used as i32 - old_end) as usize;
            let mut data_after = vec![0u8; data_after_len];

            let blob_r = conn
                .blob_open(
                    rusqlite::DatabaseName::Main,
                    "col_chunks",
                    "data",
                    rowid,
                    true, // read-only
                )
                .map_err(ColError::from)?;

            blob_r
                .read_at(&mut data_after, old_end as usize)
                .map_err(ColError::from)?;
            drop(blob_r);

            // Write new data + shifted data
            let mut blob_w = conn
                .blob_open(
                    rusqlite::DatabaseName::Main,
                    "col_chunks",
                    "data",
                    rowid,
                    false, // writable
                )
                .map_err(ColError::from)?;

            blob_w
                .write_at(new_data_bytes, old_start as usize)
                .map_err(ColError::from)?;
            blob_w
                .write_at(&data_after, (old_start + new_size) as usize)
                .map_err(ColError::from)?;

            drop(blob_w);

            // Update THIS element's index
            let new_index = pack_index_triple(chunk_id, old_start, old_start + new_size);
            Self::update_index_entry_internal(
                &conn,
                idx_col_id,
                elem_idx,
                &new_index,
                max_chunk_bytes,
            )?;

            // Shift all subsequent elements' indices in this chunk
            Self::shift_chunk_indices_after(
                &conn,
                idx_col_id,
                elem_idx + 1,
                chunk_id,
                size_delta,
                max_chunk_bytes,
            )?;

            // Update bytes_used (grew by delta)
            let new_bytes_used = bytes_used + size_delta as i64;
            conn.execute(
                "UPDATE col_chunks SET bytes_used = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                params![new_bytes_used, data_col_id, chunk_id as i64],
            )
            .map_err(ColError::from)?;

            return Ok((chunk_id, old_start, old_start + new_size));
        }

        // CASE 3: Doesn't fit - rebuild chunk
        // Get all elements in this chunk
        let chunk_elements =
            Self::get_chunk_elements(&conn, idx_col_id, chunk_id, max_chunk_bytes)?;

        // Build new chunk data by reading all elements and replacing the updated one
        let mut all_data = Vec::new();
        let mut all_indices = Vec::new();

        for (e_idx, start, end) in chunk_elements.iter() {
            if *e_idx == elem_idx {
                // Add NEW data for this element
                all_data.push(new_data_bytes.to_vec());
            } else {
                // Read existing data
                let data =
                    Self::read_element_from_chunk(&conn, data_col_id, chunk_id, *start, *end)?;
                all_data.push(data);
            }
        }

        // Calculate total size
        let total_size: i64 = all_data.iter().map(|d| d.len() as i64).sum();

        // Get min_chunk_bytes for sizing
        let min_chunk_bytes: i64 = conn
            .query_row(
                "SELECT min_chunk_bytes FROM col_meta WHERE col_id = ?1",
                params![data_col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Check if chunk would be too large (> 2x max) - need to split
        if total_size > max_chunk_bytes * 2 {
            return Err(ColError::Col(format!(
                "Update would create chunk larger than 2x max ({}MB > {}MB). \
                 Chunk splitting not yet implemented in single setitem. \
                 Consider using slice operations or vacuum.",
                total_size / 1024 / 1024,
                (max_chunk_bytes * 2) / 1024 / 1024
            ))
            .into());
        }

        // Find appropriate new chunk size with headroom
        let mut new_capacity = min_chunk_bytes;
        while new_capacity < total_size {
            new_capacity *= 2;
        }

        // Cap at max, but allow up to 1.5x max for small overflows
        if total_size > max_chunk_bytes {
            if total_size <= max_chunk_bytes * 3 / 2 {
                new_capacity = total_size; // Small overflow, use exact size
            } else {
                new_capacity = max_chunk_bytes; // Large, cap at max
            }
        } else {
            // CRITICAL: Keep the power-of-2 size for headroom!
            // Don't use exact total_size or next update will trigger rebuild again
            new_capacity = std::cmp::min(new_capacity, max_chunk_bytes);
        }

        // Rewrite chunk with new data
        // CRITICAL FIX: Use zeroblob() and BLOB API to avoid huge Vec allocation
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![data_col_id, chunk_id as i64],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        // Update chunk size using zeroblob
        conn.execute(
            "UPDATE col_chunks SET data = zeroblob(?1), actual_size = ?1, bytes_used = ?2
             WHERE col_id = ?3 AND chunk_idx = ?4",
            params![new_capacity, total_size, data_col_id, chunk_id as i64],
        )
        .map_err(ColError::from)?;

        // Open blob for writing
        let mut blob = conn
            .blob_open(
                rusqlite::DatabaseName::Main,
                "col_chunks",
                "data",
                rowid,
                false, // writable
            )
            .map_err(ColError::from)?;

        // Write all elements sequentially using BLOB API
        let mut offset = 0usize;
        for (i, data) in all_data.iter().enumerate() {
            blob.write_at(data, offset).map_err(ColError::from)?;

            let e_idx = chunk_elements[i].0;
            let start = offset as i32;
            let end = (offset + data.len()) as i32;

            all_indices.push((e_idx, start, end));
            offset += data.len();
        }

        drop(blob);

        // Update ALL index entries for this chunk
        for (e_idx, start, end) in all_indices.iter() {
            let new_index = pack_index_triple(chunk_id, *start, *end);
            Self::update_index_entry_internal(
                &conn,
                idx_col_id,
                *e_idx,
                &new_index,
                max_chunk_bytes,
            )?;
        }

        // Return new position for the updated element
        let updated_entry = all_indices
            .iter()
            .find(|(e_idx, _, _)| *e_idx == elem_idx)
            .unwrap();

        Ok((chunk_id, updated_entry.1, updated_entry.2))
    }

    /// Update multiple variable-size elements via slice (v0.5.0).
    ///
    /// Strategy:
    /// - If total_new_bytes ≤ total_old_bytes: Direct replace, update indices, DON'T touch bytes_used
    /// - If total_new_bytes > total_old_bytes: Rebuild affected chunks, update bytes_used, split if > 2x max
    ///
    /// # Arguments
    /// * `data_col_id` - Data column ID
    /// * `idx_col_id` - Index column ID
    /// * `start_idx` - Starting element index
    /// * `count` - Number of elements to update
    /// * `new_values` - List of new values (PyBytes)
    /// * `max_chunk_bytes` - Max chunk size
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_varsize_slice_impl(
        &self,
        _py: Python<'_>,
        data_col_id: i64,
        idx_col_id: i64,
        start_idx: i64,
        count: i64,
        new_values: &Bound<'_, PyList>,
        max_chunk_bytes: i64,
    ) -> PyResult<()> {
        if count == 0 {
            return Ok(());
        }

        let conn = self.conn.lock().unwrap();

        // 1. Read all affected index entries
        let index_data =
            self._read_range_internal(&conn, idx_col_id, start_idx, count, 12, max_chunk_bytes)?;

        // Parse old indices and calculate total old size
        let mut old_indices: Vec<(i64, i32, i32, i32)> = Vec::new(); // (elem_idx, chunk_id, start, end)
        let mut total_old_size = 0i64;

        for i in 0..count as usize {
            let offset = i * 12;
            let chunk_id = i32::from_le_bytes([
                index_data[offset],
                index_data[offset + 1],
                index_data[offset + 2],
                index_data[offset + 3],
            ]);
            let start = i32::from_le_bytes([
                index_data[offset + 4],
                index_data[offset + 5],
                index_data[offset + 6],
                index_data[offset + 7],
            ]);
            let end = i32::from_le_bytes([
                index_data[offset + 8],
                index_data[offset + 9],
                index_data[offset + 10],
                index_data[offset + 11],
            ]);

            old_indices.push((start_idx + i as i64, chunk_id, start, end));
            total_old_size += (end - start) as i64;
        }

        // Calculate total new size
        let mut new_values_vec: Vec<Vec<u8>> = Vec::new();
        let mut total_new_size = 0i64;

        for i in 0..count as usize {
            let val = new_values
                .get_item(i)?
                .downcast::<PyBytes>()?
                .as_bytes()
                .to_vec();
            total_new_size += val.len() as i64;
            new_values_vec.push(val);
        }

        // 2. Decide strategy based on total size comparison
        if total_new_size <= total_old_size {
            // DIRECT REPLACEMENT MODE - write in place, leave fragments
            Self::update_slice_direct_mode(
                &conn,
                data_col_id,
                idx_col_id,
                &old_indices,
                &new_values_vec,
                max_chunk_bytes,
            )?;
        } else {
            // REBUILD MODE - reconstruct affected chunks
            Self::update_slice_rebuild_mode(
                &conn,
                data_col_id,
                idx_col_id,
                &old_indices,
                &new_values_vec,
                max_chunk_bytes,
            )?;
        }

        Ok(())
    }

    /// Delete element from variable-size column (marks chunk as having deletions).
    ///
    /// Note: Doesn't actually remove data, just marks for vacuum
    pub(crate) fn delete_adaptive_impl(&self, idx_col_id: i64, elem_idx: i64) -> PyResult<i32> {
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
}

// Helper methods (not exposed to Python)
impl super::_ColumnVault {
    /// Update a single index entry efficiently using BLOB API.
    ///
    /// Index entries are 12-byte triples stored in index column chunks.
    pub(crate) fn update_index_entry_internal(
        conn: &Connection,
        idx_col_id: i64,
        elem_idx: i64,
        new_index: &[u8; 12],
        idx_max_chunk: i64,
    ) -> Result<(), ColError> {
        // Index column is fixed-size (12 bytes per entry)
        // Use the same logic as write_range for proper chunk handling
        let start_byte = elem_idx * 12;
        let end_byte = start_byte + 12;

        // Calculate which chunk(s) this spans
        let start_chunk = start_byte / idx_max_chunk;
        let end_chunk = (end_byte - 1) / idx_max_chunk;

        // For a 12-byte write, it should only span 1 chunk unless at boundary
        for chunk_idx in start_chunk..=end_chunk {
            let chunk_start_byte = chunk_idx * idx_max_chunk;
            let chunk_end_byte = chunk_start_byte + idx_max_chunk;

            let write_start = std::cmp::max(start_byte, chunk_start_byte);
            let write_end = std::cmp::min(end_byte, chunk_end_byte);
            let bytes_to_write = (write_end - write_start) as usize;

            if bytes_to_write == 0 {
                continue;
            }

            let offset_in_chunk = (write_start - chunk_start_byte) as usize;
            let offset_in_data = (write_start - start_byte) as usize;

            // Ensure chunk exists (create if needed)
            let chunk_exists: bool = conn
                .query_row(
                    "SELECT 1 FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![idx_col_id, chunk_idx],
                    |_| Ok(true),
                )
                .unwrap_or(false);

            if !chunk_exists {
                // Create chunk at max size for random access
                // CRITICAL FIX: Use SQLite's zeroblob() to avoid memory allocation
                conn.execute(
                    "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size)
                     VALUES (?1, ?2, zeroblob(?3), ?3)",
                    params![idx_col_id, chunk_idx, idx_max_chunk],
                )
                .map_err(ColError::from)?;
            }

            // CRITICAL FIX: Use BLOB API instead of read-modify-write pattern
            // This is called in a loop for variable-size updates, so performance is critical!

            // Check current chunk size
            let (rowid, actual_size): (i64, i64) = conn
                .query_row(
                    "SELECT rowid, actual_size FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![idx_col_id, chunk_idx],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(ColError::from)?;

            // Grow chunk if needed
            let needed_size = offset_in_chunk + bytes_to_write;
            if needed_size > actual_size as usize {
                // Grow to idx_max_chunk
                conn.execute(
                    "UPDATE col_chunks SET data = zeroblob(?1), actual_size = ?1
                     WHERE col_id = ?2 AND chunk_idx = ?3",
                    params![idx_max_chunk, idx_col_id, chunk_idx],
                )
                .map_err(ColError::from)?;
            }

            // Write using BLOB API (only writes the 12 bytes we need!)
            let mut blob = conn
                .blob_open(
                    rusqlite::DatabaseName::Main,
                    "col_chunks",
                    "data",
                    rowid,
                    false, // writable
                )
                .map_err(ColError::from)?;

            blob.write_at(
                &new_index[offset_in_data..offset_in_data + bytes_to_write],
                offset_in_chunk,
            )
            .map_err(ColError::from)?;
        }

        Ok(())
    }

    /// Shift start/end bytes for all index entries after elem_idx in the same chunk.
    ///
    /// OPTIMIZED: Reads index chunks in batches, only updates affected ones.
    pub(crate) fn shift_chunk_indices_after(
        conn: &Connection,
        idx_col_id: i64,
        start_elem_idx: i64,
        target_chunk_id: i32,
        delta: i32,
        idx_max_chunk: i64,
    ) -> Result<(), ColError> {
        let length: i64 = conn
            .query_row(
                "SELECT length FROM col_meta WHERE col_id = ?1",
                params![idx_col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        if start_elem_idx >= length {
            return Ok(());
        }

        // Determine which index chunks we need to process
        let start_byte = start_elem_idx * 12;
        let end_byte = length * 12;
        let start_chunk = start_byte / idx_max_chunk;
        let end_chunk = (end_byte - 1) / idx_max_chunk;

        // Process each index chunk that might contain affected elements
        for idx_chunk_idx in start_chunk..=end_chunk {
            // Read this index chunk
            let chunk_data: Vec<u8> = match conn.query_row(
                "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![idx_col_id, idx_chunk_idx],
                |row| row.get(0),
            ) {
                Ok(data) => data,
                Err(_) => continue, // Chunk doesn't exist, skip
            };

            let mut modified = false;
            let mut new_chunk_data = chunk_data.clone();

            // Calculate elem_idx range for this chunk
            let chunk_start_byte = idx_chunk_idx * idx_max_chunk;
            let first_elem_in_chunk = chunk_start_byte / 12;
            let num_entries = chunk_data.len() / 12;

            for i in 0..num_entries {
                let elem_idx = first_elem_in_chunk + i as i64;

                if elem_idx < start_elem_idx || elem_idx >= length {
                    continue;
                }

                let offset = i * 12;

                let cid = i32::from_le_bytes([
                    chunk_data[offset],
                    chunk_data[offset + 1],
                    chunk_data[offset + 2],
                    chunk_data[offset + 3],
                ]);

                // Only update if in target chunk
                if cid == target_chunk_id {
                    let start = i32::from_le_bytes([
                        chunk_data[offset + 4],
                        chunk_data[offset + 5],
                        chunk_data[offset + 6],
                        chunk_data[offset + 7],
                    ]);
                    let end = i32::from_le_bytes([
                        chunk_data[offset + 8],
                        chunk_data[offset + 9],
                        chunk_data[offset + 10],
                        chunk_data[offset + 11],
                    ]);

                    // Shift both start and end
                    let new_start = start + delta;
                    let new_end = end + delta;
                    let new_index = pack_index_triple(cid, new_start, new_end);

                    // Update in buffer
                    new_chunk_data[offset..offset + 12].copy_from_slice(&new_index);
                    modified = true;
                }
            }

            // Write back only if modified
            if modified {
                conn.execute(
                    "UPDATE col_chunks SET data = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                    params![new_chunk_data, idx_col_id, idx_chunk_idx],
                )
                .map_err(ColError::from)?;
            }
        }

        Ok(())
    }

    /// Get all elements in a specific data chunk by scanning the index column.
    /// Returns Vec<(elem_idx, start_byte, end_byte)> sorted by start_byte.
    ///
    /// OPTIMIZED: Reads all index chunks once, parses in memory.
    pub(crate) fn get_chunk_elements(
        conn: &Connection,
        idx_col_id: i64,
        target_chunk_id: i32,
        _idx_max_chunk: i64,
    ) -> Result<Vec<(i64, i32, i32)>, ColError> {
        let length: i64 = conn
            .query_row(
                "SELECT length FROM col_meta WHERE col_id = ?1",
                params![idx_col_id],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        if length == 0 {
            return Ok(Vec::new());
        }

        let mut elements = Vec::new();

        // Read ALL index chunks at once
        let mut stmt = conn
            .prepare("SELECT chunk_idx, data FROM col_chunks WHERE col_id = ?1 ORDER BY chunk_idx")
            .map_err(ColError::from)?;

        let chunks = stmt
            .query_map(params![idx_col_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(ColError::from)?;

        let mut elem_idx = 0i64;

        for chunk_result in chunks {
            let (_chunk_idx, chunk_data) = chunk_result.map_err(ColError::from)?;

            // Parse all 12-byte entries in this chunk
            let num_entries = chunk_data.len() / 12;

            for i in 0..num_entries {
                if elem_idx >= length {
                    break;
                }

                let offset = i * 12;

                let cid = i32::from_le_bytes([
                    chunk_data[offset],
                    chunk_data[offset + 1],
                    chunk_data[offset + 2],
                    chunk_data[offset + 3],
                ]);

                if cid == target_chunk_id {
                    let start = i32::from_le_bytes([
                        chunk_data[offset + 4],
                        chunk_data[offset + 5],
                        chunk_data[offset + 6],
                        chunk_data[offset + 7],
                    ]);
                    let end = i32::from_le_bytes([
                        chunk_data[offset + 8],
                        chunk_data[offset + 9],
                        chunk_data[offset + 10],
                        chunk_data[offset + 11],
                    ]);

                    elements.push((elem_idx, start, end));
                }

                elem_idx += 1;
            }
        }

        // Sort by start_byte to maintain order
        elements.sort_by_key(|(_, start, _)| *start);

        Ok(elements)
    }

    /// Read element data from chunk given start/end bytes.
    pub(crate) fn read_element_from_chunk(
        conn: &Connection,
        data_col_id: i64,
        chunk_id: i32,
        start_byte: i32,
        end_byte: i32,
    ) -> Result<Vec<u8>, ColError> {
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![data_col_id, chunk_id as i64],
                |row| row.get(0),
            )
            .map_err(ColError::from)?;

        let blob = conn
            .blob_open(
                rusqlite::DatabaseName::Main,
                "col_chunks",
                "data",
                rowid,
                true, // read-only
            )
            .map_err(ColError::from)?;

        let len = (end_byte - start_byte) as usize;
        let mut result = vec![0u8; len];
        blob.read_at(&mut result, start_byte as usize)
            .map_err(ColError::from)?;

        Ok(result)
    }

    /// Direct replacement mode for slice update (total_new ≤ total_old).
    /// Writes new data sequentially, updates indices, leaves fragments.
    ///
    /// IMPORTANT: Can't write at old positions independently because elements are contiguous!
    /// Must rebuild the sequence starting from first updated element.
    pub(crate) fn update_slice_direct_mode(
        conn: &Connection,
        data_col_id: i64,
        idx_col_id: i64,
        old_indices: &[(i64, i32, i32, i32)], // (elem_idx, chunk_id, start, end)
        new_values: &[Vec<u8>],
        idx_max_chunk: i64,
    ) -> Result<(), ColError> {
        // Create update map
        let update_map: HashMap<i64, &Vec<u8>> = old_indices
            .iter()
            .enumerate()
            .map(|(i, (elem_idx, _, _, _))| (*elem_idx, &new_values[i]))
            .collect();

        // Group old_indices by chunk
        let mut chunks_affected: HashMap<i32, Vec<i64>> = HashMap::new();
        for (elem_idx, chunk_id, _, _) in old_indices {
            chunks_affected
                .entry(*chunk_id)
                .or_default()
                .push(*elem_idx);
        }

        // Process each affected chunk
        for (chunk_id, _) in chunks_affected {
            // Get elements we're updating in this chunk from old_indices
            let mut chunk_updates: Vec<(i64, i32, i32, &Vec<u8>)> = Vec::new();

            for (elem_idx, cid, old_start, old_end) in old_indices {
                if *cid == chunk_id {
                    let new_val = update_map.get(elem_idx).unwrap();
                    chunk_updates.push((*elem_idx, *old_start, *old_end, new_val));
                }
            }

            if chunk_updates.is_empty() {
                continue;
            }

            // Sort by old_start to process in order
            chunk_updates.sort_by_key(|(_, start, _, _)| *start);

            // For direct mode (total_new ≤ total_old):
            // Build contiguous new data, write sequentially starting from first element
            let write_start = chunk_updates.first().unwrap().1;
            let mut new_section = Vec::new();
            let mut new_indices = Vec::new();

            for (elem_idx, _old_start, _old_end, new_val) in chunk_updates {
                let start = write_start + new_section.len() as i32;
                new_section.extend_from_slice(new_val);
                let end = write_start + new_section.len() as i32;
                new_indices.push((elem_idx, start, end));
            }

            // Write the new section
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![data_col_id, chunk_id as i64],
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

            blob.write_at(&new_section, write_start as usize)
                .map_err(ColError::from)?;
            drop(blob);

            // Batch update indices (much faster than one-by-one)
            Self::batch_update_index_entries(
                conn,
                idx_col_id,
                &new_indices
                    .iter()
                    .map(|(elem_idx, start, end)| {
                        (*elem_idx, pack_index_triple(chunk_id, *start, *end))
                    })
                    .collect::<Vec<_>>(),
                idx_max_chunk,
            )?;

            // IMPORTANT: DON'T update bytes_used (fragments remain for vacuum!)
        }

        Ok(())
    }

    /// Batch update multiple index entries efficiently.
    /// Groups updates by index chunk to minimize I/O.
    pub(crate) fn batch_update_index_entries(
        conn: &Connection,
        idx_col_id: i64,
        updates: &[(i64, [u8; 12])], // (elem_idx, new_index_data)
        idx_max_chunk: i64,
    ) -> Result<(), ColError> {
        // Group updates by which index chunk they're in
        let mut by_idx_chunk: HashMap<i64, Vec<(usize, i64)>> = HashMap::new();

        for (i, (elem_idx, _)) in updates.iter().enumerate() {
            let byte_offset = elem_idx * 12;
            let idx_chunk_idx = byte_offset / idx_max_chunk;
            by_idx_chunk
                .entry(idx_chunk_idx)
                .or_default()
                .push((i, *elem_idx));
        }

        // Process each index chunk
        for (idx_chunk_idx, entries) in by_idx_chunk {
            // Read index chunk once
            let mut chunk_data: Vec<u8> = conn
                .query_row(
                    "SELECT data FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![idx_col_id, idx_chunk_idx],
                    |row| row.get(0),
                )
                .map_err(ColError::from)?;

            // Update all entries in this chunk
            for (update_idx, elem_idx) in entries {
                let byte_offset = elem_idx * 12;
                let offset_in_chunk = (byte_offset % idx_max_chunk) as usize;
                let new_index = &updates[update_idx].1;

                chunk_data[offset_in_chunk..offset_in_chunk + 12].copy_from_slice(new_index);
            }

            // Write back once
            conn.execute(
                "UPDATE col_chunks SET data = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                params![chunk_data, idx_col_id, idx_chunk_idx],
            )
            .map_err(ColError::from)?;
        }

        Ok(())
    }

    /// Rebuild mode for slice update (total_new > total_old).
    /// Rebuilds affected chunks, updates bytes_used, splits if needed.
    pub(crate) fn update_slice_rebuild_mode(
        conn: &Connection,
        data_col_id: i64,
        idx_col_id: i64,
        old_indices: &[(i64, i32, i32, i32)],
        new_values: &[Vec<u8>],
        max_chunk_bytes: i64,
    ) -> Result<(), ColError> {
        // Create map for quick lookup of which elements are being updated
        let update_map: HashMap<i64, &Vec<u8>> = old_indices
            .iter()
            .enumerate()
            .map(|(i, (elem_idx, _, _, _))| (*elem_idx, &new_values[i]))
            .collect();

        // Group old_indices by chunk
        let mut chunks_affected: HashMap<i32, Vec<i64>> = HashMap::new();
        for (elem_idx, chunk_id, _, _) in old_indices {
            chunks_affected
                .entry(*chunk_id)
                .or_default()
                .push(*elem_idx);
        }

        // Process each affected chunk
        for (chunk_id, _) in chunks_affected {
            // Get ALL elements in this chunk (not just updated ones)
            let all_elements =
                Self::get_chunk_elements(conn, idx_col_id, chunk_id, max_chunk_bytes)?;

            // Build new chunk data
            let mut new_chunk_data = Vec::new();
            let mut new_indices_for_chunk: Vec<(i64, i32, i32)> = Vec::new();

            for (elem_idx, old_start, old_end) in all_elements {
                let start = new_chunk_data.len() as i32;

                if let Some(new_data) = update_map.get(&elem_idx) {
                    // Use new value
                    new_chunk_data.extend_from_slice(new_data);
                } else {
                    // Keep old value
                    let old_data = Self::read_element_from_chunk(
                        conn,
                        data_col_id,
                        chunk_id,
                        old_start,
                        old_end,
                    )?;
                    new_chunk_data.extend_from_slice(&old_data);
                }

                let end = new_chunk_data.len() as i32;
                new_indices_for_chunk.push((elem_idx, start, end));
            }

            let new_size = new_chunk_data.len() as i64;

            // Check if chunk needs splitting (> 2x max)
            if new_size > max_chunk_bytes * 2 {
                return Err(ColError::Col(format!(
                    "Slice update would create chunk larger than 2x max ({}MB > {}MB). \
                     Chunk splitting not yet fully implemented. Consider smaller updates.",
                    new_size / 1024 / 1024,
                    (max_chunk_bytes * 2) / 1024 / 1024
                ))
                .into());
            }

            // Get min_chunk_bytes and current actual_size
            let (min_chunk_bytes, actual_size): (i64, i64) = conn
                .query_row(
                    "SELECT min_chunk_bytes, actual_size FROM col_meta cm
                     JOIN col_chunks cc ON cm.col_id = cc.col_id
                     WHERE cc.col_id = ?1 AND cc.chunk_idx = ?2",
                    params![data_col_id, chunk_id as i64],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(ColError::from)?;

            // Find appropriate capacity - use smallest power of 2 that fits
            let mut new_capacity = min_chunk_bytes;
            while new_capacity < new_size {
                new_capacity *= 2;
            }

            // Cap at reasonable limits
            if new_size <= max_chunk_bytes {
                // Fits in max, use the calculated power of 2
                new_capacity = std::cmp::min(new_capacity, max_chunk_bytes);
            } else if new_size <= max_chunk_bytes * 3 / 2 {
                // Small overflow, allow exact size up to 1.5x max
                new_capacity = new_size;
            } else {
                // Large overflow - use max (will be capped at 2x max by check above)
                new_capacity = max_chunk_bytes;
            }

            // Resize chunk if needed
            if new_capacity > actual_size {
                conn.execute(
                    "UPDATE col_chunks SET data = zeroblob(?1), actual_size = ?1
                     WHERE col_id = ?2 AND chunk_idx = ?3",
                    params![new_capacity, data_col_id, chunk_id as i64],
                )
                .map_err(ColError::from)?;
            }

            // Write new chunk data
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                    params![data_col_id, chunk_id as i64],
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

            blob.write_at(&new_chunk_data, 0).map_err(ColError::from)?;
            drop(blob);

            // Update bytes_used
            conn.execute(
                "UPDATE col_chunks SET bytes_used = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
                params![new_size, data_col_id, chunk_id as i64],
            )
            .map_err(ColError::from)?;

            // Update ALL index entries for this chunk
            for (elem_idx, start, end) in new_indices_for_chunk {
                let new_index = pack_index_triple(chunk_id, start, end);
                Self::update_index_entry_internal(
                    conn,
                    idx_col_id,
                    elem_idx,
                    &new_index,
                    max_chunk_bytes,
                )?;
            }
        }

        Ok(())
    }
}
