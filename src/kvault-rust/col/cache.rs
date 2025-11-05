// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Column-specific caching for append/extend operations.
//!
//! ColumnCache is different from KVault's WriteBackCache:
//! - Stores append-only buffers (not key-value pairs)
//! - Supports both fixed-size and variable-size columns
//! - Handles index column coordination for variable-size data

use std::time::Instant;

/// Cache for column append/extend operations.
/// Similar to KVault's WriteBackCache but adapted for columnar storage.
pub(crate) struct ColumnCache {
    /// Buffer for fixed-size columns (concatenated packed bytes)
    pub fixed_buffer: Vec<u8>,
    /// Buffer for variable-size columns (individual elements)
    pub var_buffer: Vec<Vec<u8>>,
    /// Current bytes in cache
    pub current_bytes: usize,
    /// Cache capacity in bytes
    pub cap_bytes: usize,
    /// Auto-flush threshold in bytes
    pub flush_threshold: usize,
    /// Last write time for daemon tracking
    pub last_write_time: Instant,
    /// Whether this is a variable-size column cache
    pub is_variable_size: bool,
    /// For variable-size columns: the index column ID
    pub idx_col_id: Option<i64>,
}

impl ColumnCache {
    /// Create a new column cache.
    pub fn new(
        cap_bytes: usize,
        flush_threshold: usize,
        is_variable_size: bool,
        idx_col_id: Option<i64>,
    ) -> Self {
        Self {
            fixed_buffer: Vec::new(),
            var_buffer: Vec::new(),
            current_bytes: 0,
            cap_bytes,
            flush_threshold,
            last_write_time: Instant::now(),
            is_variable_size,
            idx_col_id,
        }
    }

    /// Check if cache needs flushing based on threshold.
    pub fn needs_flush(&self) -> bool {
        self.current_bytes >= self.flush_threshold
    }

    /// Check if adding data would exceed capacity.
    pub fn would_exceed_capacity(&self, additional_bytes: usize) -> bool {
        self.current_bytes + additional_bytes > self.cap_bytes
    }

    /// Add data to cache. Returns true if cache needs flushing.
    pub fn append(&mut self, data: Vec<u8>) -> bool {
        let data_size = data.len();

        if self.is_variable_size {
            self.var_buffer.push(data);
        } else {
            self.fixed_buffer.extend_from_slice(&data);
        }

        self.current_bytes += data_size;
        self.last_write_time = Instant::now();

        self.needs_flush()
    }

    /// Add multiple elements to cache (for extend operations).
    pub fn extend(&mut self, elements: Vec<Vec<u8>>) -> bool {
        for elem in elements {
            let elem_size = elem.len();

            if self.is_variable_size {
                self.var_buffer.push(elem);
            } else {
                self.fixed_buffer.extend_from_slice(&elem);
            }

            self.current_bytes += elem_size;
        }

        self.last_write_time = Instant::now();
        self.needs_flush()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_bytes == 0
    }

    /// Get idle time since last write.
    pub fn idle_time(&self) -> std::time::Duration {
        self.last_write_time.elapsed()
    }

    /// Clear the cache and return the buffered data.
    pub fn take(&mut self) -> (Vec<u8>, Vec<Vec<u8>>) {
        let fixed = std::mem::take(&mut self.fixed_buffer);
        let var = std::mem::take(&mut self.var_buffer);
        self.current_bytes = 0;
        (fixed, var)
    }
}
