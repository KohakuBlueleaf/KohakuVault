// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Common infrastructure shared across KVault and ColumnVault.
//!
//! This module provides:
//! - Unified error types
//! - Generic write-back cache
//! - SQLite connection helpers (WAL, pragmas, checkpointing)
//! - BLOB API helpers
//! - Metadata table management

pub mod meta;

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use rusqlite::blob::Blob;
use rusqlite::{Connection, OpenFlags};
use thiserror::Error;

// ========================================
// ERROR TYPES
// ========================================

/// Unified error type for all vault operations.
/// Replaces separate KvError and ColError types.
#[derive(Error, Debug)]
pub enum VaultError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Python error: {0}")]
    Py(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Column error: {0}")]
    Column(String),
}

impl From<VaultError> for PyErr {
    fn from(e: VaultError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

// ========================================
// GENERIC WRITE-BACK CACHE
// ========================================

/// Generic write-back cache with capacity management.
/// Used by KVault for key-value caching.
pub struct WriteBackCache<K, V> {
    pub map: HashMap<K, V>,
    pub current_bytes: usize,
    pub cap_bytes: usize,
    pub flush_threshold: usize,
    pub last_write_time: Option<Instant>,
}

impl<K, V> WriteBackCache<K, V>
where
    K: std::hash::Hash + Eq,
{
    pub fn new(cap_bytes: usize, flush_threshold: usize) -> Self {
        Self {
            map: HashMap::new(),
            current_bytes: 0,
            cap_bytes,
            flush_threshold,
            last_write_time: None,
        }
    }

    /// Check if cache should be flushed based on threshold.
    pub fn should_flush(&self) -> bool {
        self.current_bytes >= self.flush_threshold
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl WriteBackCache<Vec<u8>, Vec<u8>> {
    /// Try to insert into cache with capacity checks.
    /// Returns error if value too large or cache needs flush first.
    pub fn insert(&mut self, k: Vec<u8>, v: Vec<u8>) -> Result<(), CacheError> {
        let value_size = k.len() + v.len();

        // Check if value is larger than cache capacity (can't cache it)
        if value_size > self.cap_bytes {
            return Err(CacheError::ValueTooLarge);
        }

        // Check if adding would exceed capacity (need flush first)
        if self.current_bytes + value_size > self.cap_bytes {
            return Err(CacheError::NeedFlush);
        }

        // Safe to insert
        self.current_bytes += value_size;
        self.map.insert(k, v);
        self.last_write_time = Some(Instant::now());

        Ok(())
    }

    /// Drain all entries from cache and reset counters.
    pub fn drain(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::with_capacity(self.map.len());
        for (k, v) in self.map.drain() {
            out.push((k, v));
        }
        self.current_bytes = 0;
        self.last_write_time = None;
        out
    }
}

/// Cache operation errors.
#[derive(Debug)]
pub enum CacheError {
    ValueTooLarge, // Value larger than cache capacity
    NeedFlush,     // Cache would exceed capacity, need flush first
}

// ========================================
// CONNECTION HELPERS
// ========================================

/// Open SQLite connection with standard configuration.
///
/// # Arguments
/// * `path` - Database file path
/// * `enable_wal` - Enable WAL mode for concurrent access
/// * `page_size` - Page size in bytes (only applies to new databases)
/// * `mmap_size` - Memory-mapped I/O size in bytes
/// * `cache_kb` - Cache size in kilobytes
///
/// # Returns
/// Configured SQLite connection
pub fn open_connection(
    path: &str,
    enable_wal: bool,
    page_size: u32,
    mmap_size: u64,
    cache_kb: i64,
) -> Result<Connection, VaultError> {
    let is_new = !Path::new(path).exists();

    // Create parent directories if needed
    if let Some(dir) = Path::new(path).parent() {
        fs::create_dir_all(dir).ok();
    }

    // Open connection with read-write-create flags
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_URI;

    let conn = Connection::open_with_flags(path, flags)?;

    // Set page size (only works on empty database)
    if is_new {
        let _ = conn.execute_batch(&format!("PRAGMA page_size={};", page_size));
    }

    // Enable WAL mode
    if enable_wal {
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        // Set WAL auto-checkpoint to 1000 pages (default)
        // This prevents WAL from growing indefinitely
        let _ = conn.pragma_update(None, "wal_autocheckpoint", 1000);
    }

    // Performance pragmas
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    let _ = conn.pragma_update(None, "mmap_size", mmap_size);
    // Negative cache_size means KB units
    let _ = conn.pragma_update(None, "cache_size", -cache_kb);
    let _ = conn.pragma_update(None, "temp_store", "MEMORY");

    // Set busy timeout to prevent database locked errors
    conn.busy_timeout(Duration::from_millis(5000))?;

    Ok(conn)
}

/// Manually checkpoint WAL file to main database.
///
/// This helps prevent WAL from growing too large.
/// Uses PASSIVE mode which is non-blocking.
///
/// # Arguments
/// * `conn` - Database connection
///
/// # Returns
/// Success indicator (0 on success)
pub fn checkpoint_wal(conn: &Connection) -> Result<i64, VaultError> {
    conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
    Ok(0)
}

// ========================================
// BLOB HELPERS
// ========================================

// Note: zeroblob is typically used inline with SQL: zeroblob(?1)
// BLOB operations use rusqlite's blob API directly (blob_open, read_at, write_at)

/// Open a BLOB for incremental I/O.
///
/// # Arguments
/// * `conn` - Database connection
/// * `table` - Table name
/// * `column` - Column name
/// * `rowid` - Row ID
/// * `read_only` - true for read-only access, false for read-write
///
/// # Returns
/// BLOB handle for incremental operations
pub fn open_blob<'conn>(
    conn: &'conn Connection,
    table: &str,
    column: &str,
    rowid: i64,
    read_only: bool,
) -> Result<Blob<'conn>, VaultError> {
    let blob = conn.blob_open(rusqlite::DatabaseName::Main, table, column, rowid, read_only)?;
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache: WriteBackCache<Vec<u8>, Vec<u8>> = WriteBackCache::new(1000, 500);
        assert!(cache.is_empty());
        assert!(!cache.should_flush());
    }

    #[test]
    fn test_cache_insert() {
        let mut cache: WriteBackCache<Vec<u8>, Vec<u8>> = WriteBackCache::new(1000, 500);

        // Insert small value - should succeed
        assert!(cache.insert(b"key1".to_vec(), b"value1".to_vec()).is_ok());
        assert!(!cache.is_empty());
        assert_eq!(cache.map.len(), 1);
    }

    #[test]
    fn test_cache_too_large() {
        let mut cache: WriteBackCache<Vec<u8>, Vec<u8>> = WriteBackCache::new(100, 50);

        // Try to insert value larger than capacity
        let large_value = vec![0u8; 200];
        assert!(matches!(
            cache.insert(b"key".to_vec(), large_value),
            Err(CacheError::ValueTooLarge)
        ));
    }

    #[test]
    fn test_cache_need_flush() {
        let mut cache: WriteBackCache<Vec<u8>, Vec<u8>> = WriteBackCache::new(100, 50);

        // Insert first value
        cache.insert(b"k1".to_vec(), vec![0u8; 40]).unwrap();

        // Try to insert second value that would exceed capacity
        assert!(matches!(
            cache.insert(b"k2".to_vec(), vec![0u8; 70]),
            Err(CacheError::NeedFlush)
        ));
    }

    #[test]
    fn test_cache_drain() {
        let mut cache: WriteBackCache<Vec<u8>, Vec<u8>> = WriteBackCache::new(1000, 500);

        cache.insert(b"key1".to_vec(), b"value1".to_vec()).unwrap();
        cache.insert(b"key2".to_vec(), b"value2".to_vec()).unwrap();

        let entries = cache.drain();
        assert_eq!(entries.len(), 2);
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes, 0);
    }
}
