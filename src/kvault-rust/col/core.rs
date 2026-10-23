//! Column catalog, connection ownership, and transactional publication.

use parking_lot::{Mutex, RwLock};
use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::{ColError, ColResult};
use crate::common::read_pool::ReadPool;

pub(crate) const FORMAT_VERSION: i64 = 1;
pub(crate) const MAX_BLOCK_BYTES: usize = 512 << 20;
pub(crate) type ColumnInfo = (i64, String, usize, usize, usize);

#[derive(Clone)]
pub(crate) struct Meta {
    pub id: i64,
    pub elem_size: usize,
    pub length: usize,
    pub block_bytes: usize,
}

pub(crate) fn meta(conn: &Connection, id: i64) -> ColResult<Meta> {
    let found = conn
        .query_row(
            "SELECT elem_size, length, block_bytes FROM kvault_col_columns WHERE col_id = ?1",
            [id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        )
        .optional()?;

    let (elem_size, length, block_bytes) = found.ok_or(ColError::NotFound)?;
    if elem_size < 0
        || length < 0
        || block_bytes <= 0
        || elem_size as u64 > MAX_BLOCK_BYTES as u64
        || block_bytes as u64 > MAX_BLOCK_BYTES as u64
    {
        return Err(ColError::Corrupt("invalid column metadata".into()));
    }

    Ok(Meta {
        id,
        elem_size: elem_size as usize,
        length: usize::try_from(length).map_err(|_| ColError::Bounds)?,
        block_bytes: block_bytes as usize,
    })
}

pub(crate) fn range_count(start: usize, count: Option<usize>, length: usize) -> ColResult<usize> {
    if start > length {
        return Err(ColError::Bounds);
    }

    let count = count.unwrap_or(length - start);
    if count > length - start {
        return Err(ColError::Bounds);
    }

    Ok(count)
}

pub(crate) fn new_length(length: usize, count: usize) -> ColResult<usize> {
    length
        .checked_add(count)
        .filter(|&value| value <= i64::MAX as usize)
        .ok_or_else(|| ColError::Invalid("column length overflow".into()))
}

pub(crate) struct Core {
    primary: Mutex<Option<Connection>>,
    readers: Option<ReadPool>,
    lifecycle: RwLock<()>,
    closed: AtomicBool,
    read_only: bool,
}

impl Core {
    pub(crate) fn open(path: &str, read_only: bool) -> ColResult<Self> {
        if path.is_empty() || path.starts_with("file:") {
            return Err(ColError::Invalid(
                "use a filesystem path or :memory:, not a SQLite URI".into(),
            ));
        }
        if read_only && path == ":memory:" {
            return Err(ColError::Invalid(
                "an in-memory database cannot be opened read-only".into(),
            ));
        }

        let flags = if read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
        };
        let mut conn = Connection::open_with_flags(path, flags)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "cache_size", -8192)?;
        conn.pragma_update(None, "foreign_keys", true)?;

        if !read_only {
            if path != ":memory:" {
                let mode: String = conn.pragma_query_value(None, "journal_mode", |r| r.get(0))?;
                if !mode.eq_ignore_ascii_case("wal") {
                    conn.pragma_update(None, "journal_mode", "WAL")?;
                }
            }
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            let initialized: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name = 'kvault_col_format')",
                [],
                |row| row.get(0),
            )?;
            if !initialized {
                Self::initialize(&mut conn)?;
            }
        }

        let version: i64 =
            conn.query_row("SELECT version FROM kvault_col_format WHERE singleton = 1", [], |r| {
                r.get(0)
            })?;
        if version != FORMAT_VERSION {
            return Err(ColError::Invalid(format!("unsupported column format version {version}")));
        }

        // Resolve the path once so later chdir calls cannot redirect readers.
        let readers = if path == ":memory:" {
            None
        } else {
            let path = std::fs::canonicalize(path)
                .map_err(|error| ColError::Invalid(error.to_string()))?;
            Some(ReadPool::new(path.to_string_lossy().into_owned(), 8192))
        };

        Ok(Self {
            primary: Mutex::new(Some(conn)),
            readers,
            lifecycle: RwLock::new(()),
            closed: AtomicBool::new(false),
            read_only,
        })
    }

    fn initialize(conn: &mut Connection) -> ColResult<()> {
        // Serialize first-time setup without a deferred read-to-write upgrade.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS kvault_col_format (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO kvault_col_format VALUES (1, 1);
            CREATE TABLE IF NOT EXISTS kvault_col_columns (
                col_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                dtype TEXT NOT NULL,
                elem_size INTEGER NOT NULL CHECK(elem_size >= 0),
                length INTEGER NOT NULL DEFAULT 0 CHECK(length >= 0),
                block_bytes INTEGER NOT NULL CHECK(block_bytes > 0)
            );
            CREATE TABLE IF NOT EXISTS kvault_col_blocks (
                block_id INTEGER PRIMARY KEY,
                col_id INTEGER NOT NULL REFERENCES kvault_col_columns(col_id) ON DELETE CASCADE,
                block_no INTEGER NOT NULL,
                first_row INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                used_bytes INTEGER NOT NULL,
                capacity_bytes INTEGER NOT NULL,
                row_capacity INTEGER NOT NULL,
                UNIQUE(col_id, block_no)
            );
            CREATE INDEX IF NOT EXISTS kvault_col_row_index
                ON kvault_col_blocks(col_id, first_row);
            CREATE TABLE IF NOT EXISTS kvault_col_data (
                block_id INTEGER PRIMARY KEY REFERENCES kvault_col_blocks(block_id) ON DELETE CASCADE,
                payload BLOB NOT NULL,
                offsets BLOB
            );"
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(crate) fn read<T>(
        &self,
        operation: impl FnOnce(&Connection) -> ColResult<T>,
    ) -> ColResult<T> {
        let _lifecycle = self.lifecycle.read();
        if self.closed.load(Ordering::Acquire) {
            return Err(ColError::Closed);
        }

        if let Some(readers) = &self.readers {
            return readers.read(operation);
        }

        let primary = self.primary.lock();
        let conn = primary.as_ref().ok_or(ColError::Closed)?;
        let tx = conn.unchecked_transaction()?;
        let result = operation(&tx);
        tx.rollback()?;
        result
    }

    pub(crate) fn write<T>(
        &self,
        operation: impl FnOnce(&Transaction<'_>) -> ColResult<T>,
    ) -> ColResult<T> {
        let _lifecycle = self.lifecycle.read();
        if self.closed.load(Ordering::Acquire) {
            return Err(ColError::Closed);
        }
        if self.read_only {
            return Err(ColError::ReadOnly);
        }

        // Acquire the write reservation before reading append positions.
        let mut primary = self.primary.lock();
        let conn = primary.as_mut().ok_or(ColError::Closed)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = operation(&tx)?;
        tx.commit()?;
        Ok(value)
    }

    pub(crate) fn create(
        &self,
        name: &str,
        dtype: &str,
        elem_size: usize,
        block_bytes: usize,
    ) -> ColResult<i64> {
        if name.is_empty()
            || dtype.is_empty()
            || block_bytes == 0
            || block_bytes > MAX_BLOCK_BYTES
            || elem_size > MAX_BLOCK_BYTES
        {
            return Err(ColError::Invalid("invalid column name, dtype, or block size".into()));
        }

        self.write(|tx| {
            tx.execute(
                "INSERT INTO kvault_col_columns(name, dtype, elem_size, block_bytes) VALUES (?1, ?2, ?3, ?4)",
                params![name, dtype, elem_size as i64, block_bytes as i64],
            )?;
            Ok(tx.last_insert_rowid())
        })
    }

    pub(crate) fn info(&self, name: &str) -> ColResult<ColumnInfo> {
        self.read(|conn| {
            let row: Option<(i64, String)> = conn
                .query_row(
                    "SELECT col_id, dtype FROM kvault_col_columns WHERE name = ?1",
                    [name],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let (id, dtype) = row.ok_or(ColError::NotFound)?;
            let meta = meta(conn, id)?;
            Ok((id, dtype, meta.elem_size, meta.length, meta.block_bytes))
        })
    }

    pub(crate) fn clear(&self, id: i64) -> ColResult<()> {
        self.write(|tx| {
            meta(tx, id)?;
            tx.execute("DELETE FROM kvault_col_blocks WHERE col_id = ?1", [id])?;
            tx.execute("UPDATE kvault_col_columns SET length = 0 WHERE col_id = ?1", [id])?;
            Ok(())
        })
    }

    pub(crate) fn checkpoint(&self) -> ColResult<(i64, i64, i64)> {
        let _lifecycle = self.lifecycle.read();
        if self.closed.load(Ordering::Acquire) {
            return Err(ColError::Closed);
        }
        if self.read_only {
            return Err(ColError::ReadOnly);
        }

        let primary = self.primary.lock();
        let conn = primary.as_ref().ok_or(ColError::Closed)?;
        Ok(conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?)
    }

    pub(crate) fn close(&self) {
        // Wait for active operations before dropping connections (also on Windows).
        let _lifecycle = self.lifecycle.write();
        self.closed.store(true, Ordering::Release);
        if let Some(readers) = &self.readers {
            readers.clear();
        }
        self.primary.lock().take();
    }
}
