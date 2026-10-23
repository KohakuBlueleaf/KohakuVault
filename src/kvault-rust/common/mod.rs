// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Common infrastructure shared across KVault and ColumnVault.
//!
//! This module provides:
//! - Unified error types
//! - SQLite connection helpers (WAL, pragmas, checkpointing)
//! - BLOB API helpers
//! - Metadata table management

pub mod meta;
pub(crate) mod read_pool;

use std::fs;
use std::path::Path;
use std::time::Duration;

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
    conn.busy_timeout(Duration::from_secs(5))?;

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

    Ok(conn)
}

/// Open an existing database without schema writes or journal-mode changes.
pub(crate) fn open_readonly_connection(
    path: &str,
    mmap_size: u64,
    cache_kb: i64,
) -> Result<Connection, VaultError> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(path, flags)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "mmap_size", mmap_size)?;
    conn.pragma_update(None, "cache_size", -cache_kb)?;
    Ok(conn)
}

/// Build a reader pool only for ordinary filesystem databases.
pub(crate) fn reader_pool(path: &str) -> Option<read_pool::ReadPool> {
    if path == ":memory:" || path.starts_with("file:") {
        return None;
    }
    std::fs::canonicalize(path)
        .ok()
        .map(|path| read_pool::ReadPool::new(path.to_string_lossy().into_owned(), 8192))
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

// KV cache unit tests live beside the replacement cache in kv/cache.rs.
