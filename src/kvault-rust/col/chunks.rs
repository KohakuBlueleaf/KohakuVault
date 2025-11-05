// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Chunk management for columnar storage.
//!
//! Handles:
//! - Dynamic chunk sizing with exponential growth
//! - Chunk creation and growth using BLOB API
//! - Element alignment for fixed-size columns
//! - Byte offset calculations

use pyo3::prelude::*;
use rusqlite::{params, Connection};

// Import ColError from parent module
use super::ColError;

// Helper functions (not impl block methods, just module-level functions)
/// Align chunk sizes to element boundaries (for fixed-size columns).
///
/// Ensures that elements don't cross chunk boundaries by rounding
/// min_chunk to next multiple and max_chunk to previous multiple.
pub(crate) fn align_chunk_sizes(elem_size: i64, min: i64, max: i64) -> PyResult<(i64, i64)> {
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

impl super::_ColumnVault {
    /// Prepare a chunk for appending data.
    /// Returns (chunk_idx, offset_in_chunk, chunk_capacity).
    ///
    /// Strategy:
    /// 1. Check if current chunk has space → return it
    /// 2. If not, try to DOUBLE current chunk (if < max)
    /// 3. If need new chunk: grow current to MAX, create new with nearest size
    pub(crate) fn prepare_append_chunk(
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
    ///
    /// CRITICAL FIX: Only allocate the smallest power-of-2 that fits the data.
    /// This prevents extend() from creating max-sized chunks for small data.
    pub(crate) fn calculate_new_chunk_size(
        &self,
        remaining_bytes: usize,
        min_chunk: i64,
        max_chunk: i64,
    ) -> i64 {
        let remaining = remaining_bytes as i64;

        // If remaining fits in min_chunk, just use min_chunk
        if remaining <= min_chunk {
            return min_chunk;
        }

        // Find the smallest power of 2 that fits remaining data
        let mut size = min_chunk;
        while size < remaining {
            size *= 2;
        }

        // Cap at max_chunk_bytes
        std::cmp::min(size, max_chunk)
    }

    /// Create a chunk with specific size.
    pub(crate) fn create_chunk_with_size(
        &self,
        conn: &Connection,
        col_id: i64,
        chunk_idx: i64,
        size: i64,
    ) -> PyResult<()> {
        // CRITICAL FIX: Use SQLite's zeroblob() to avoid memory allocation
        conn.execute(
            "INSERT INTO col_chunks (col_id, chunk_idx, data, actual_size) VALUES (?1, ?2, zeroblob(?3), ?3)",
            params![col_id, chunk_idx, size],
        )
        .map_err(ColError::from)?;
        Ok(())
    }

    /// Grow a chunk to specified size (e.g., double, or to max).
    /// CRITICAL: Check actual_size first to avoid unnecessary blob reads.
    pub(crate) fn grow_chunk_to_size(
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

        // CRITICAL FIX: Don't read entire blob into memory, use BLOB API
        // First, get rowid and old data size
        let (rowid, old_size): (i64, i64) = conn
            .query_row(
                "SELECT rowid, actual_size FROM col_chunks WHERE col_id = ?1 AND chunk_idx = ?2",
                params![col_id, chunk_idx],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ColError::from)?;

        // Read old data using BLOB API
        let old_blob = conn
            .blob_open(
                rusqlite::DatabaseName::Main,
                "col_chunks",
                "data",
                rowid,
                true, // readonly
            )
            .map_err(ColError::from)?;

        let mut old_data = vec![0u8; old_size as usize];
        old_blob.read_at(&mut old_data, 0).map_err(ColError::from)?;
        drop(old_blob);

        // Update chunk with new size using zeroblob, then write old data back
        conn.execute(
            "UPDATE col_chunks SET data = zeroblob(?1), actual_size = ?1 WHERE col_id = ?2 AND chunk_idx = ?3",
            params![target_size, col_id, chunk_idx],
        )
        .map_err(ColError::from)?;

        // Write old data back using BLOB API
        let mut new_blob = conn
            .blob_open(
                rusqlite::DatabaseName::Main,
                "col_chunks",
                "data",
                rowid,
                false, // writable
            )
            .map_err(ColError::from)?;

        new_blob.write_at(&old_data, 0).map_err(ColError::from)?;

        Ok(())
    }

    /// Calculate total bytes stored before a given chunk.
    pub(crate) fn get_bytes_before_chunk(
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

/// Pack 12-byte index triple (chunk_id, start_byte, end_byte).
/// Used by variable-size columns to track element locations.
pub(crate) fn pack_index_triple(chunk_id: i32, start: i32, end: i32) -> [u8; 12] {
    let mut result = [0u8; 12];
    result[0..4].copy_from_slice(&chunk_id.to_le_bytes());
    result[4..8].copy_from_slice(&start.to_le_bytes());
    result[8..12].copy_from_slice(&end.to_le_bytes());
    result
}
