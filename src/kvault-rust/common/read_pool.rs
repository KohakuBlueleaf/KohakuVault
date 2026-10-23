//! Short-lived read snapshots on independent SQLite connections.

use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags};
use std::time::Duration;

/// Retain a small number of idle readers without serializing active reads.
pub(crate) struct ReadPool {
    path: String,
    idle: Mutex<Option<Vec<Connection>>>,
    cache_kb: i64,
    mmap_size: u64,
    idle_limit: usize,
}

impl ReadPool {
    pub(crate) fn new(path: String, cache_kb: i64) -> Self {
        Self { path, idle: Mutex::new(Some(Vec::new())), cache_kb, mmap_size: 0, idle_limit: 8 }
    }

    pub(crate) fn configured(mut self, cache_kb: i64, mmap_size: u64, idle_limit: usize) -> Self {
        self.cache_kb = cache_kb;
        self.mmap_size = mmap_size;
        self.idle_limit = idle_limit;
        self
    }

    pub(crate) fn read<T, E>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<rusqlite::Error>,
    {
        self.lease(true, operation)
    }

    /// A single SELECT already owns its SQLite snapshot until its statement ends.
    pub(crate) fn query<T, E>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<rusqlite::Error>,
    {
        self.lease(false, operation)
    }

    fn lease<T, E>(
        &self,
        snapshot: bool,
        operation: impl FnOnce(&Connection) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<rusqlite::Error>,
    {
        // Do not hold the pool lock while opening or using a connection.
        let cached = self
            .idle
            .lock()
            .as_mut()
            .ok_or(rusqlite::Error::InvalidQuery)?
            .pop();
        let conn = match cached {
            Some(conn) => conn,
            None => {
                let conn =
                    Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                conn.busy_timeout(Duration::from_secs(5))?;
                conn.pragma_update(None, "cache_size", -self.cache_kb)?;
                conn.pragma_update(None, "mmap_size", self.mmap_size)?;
                conn
            }
        };

        // Metadata and payload reads must observe the same committed version.
        let (result, cleanup) = if snapshot {
            let tx = conn.unchecked_transaction()?;
            let result = operation(&tx);
            (result, tx.rollback())
        } else {
            let result = operation(&conn);
            let cleanup = if conn.is_autocommit() {
                Ok(())
            } else {
                conn.execute_batch("ROLLBACK")
            };
            (result, cleanup)
        };

        if cleanup.is_ok() {
            let mut idle = self.idle.lock();
            if let Some(idle) = idle.as_mut() {
                if idle.len() < self.idle_limit {
                    idle.push(conn);
                }
            }
        }

        let value = result?;
        cleanup?;
        Ok(value)
    }

    /// SQLite page-cache allocation of idle connections, not total process RSS.
    pub(crate) fn stats(&self) -> (usize, i64) {
        let guard = self.idle.lock();
        let idle = guard.as_deref().unwrap_or(&[]);
        let bytes = idle
            .iter()
            .map(|conn| {
                let mut current = 0;
                let mut highwater = 0;
                // Each idle connection is exclusively owned under the pool lock.
                unsafe {
                    rusqlite::ffi::sqlite3_db_status(
                        conn.handle(),
                        rusqlite::ffi::SQLITE_DBSTATUS_CACHE_USED,
                        &mut current,
                        &mut highwater,
                        0,
                    );
                }
                i64::from(current)
            })
            .sum();
        (idle.len(), bytes)
    }

    pub(crate) fn clear(&self) {
        if let Some(idle) = self.idle.lock().as_mut() {
            idle.clear();
        }
    }

    pub(crate) fn close(&self) {
        // Leased readers finish normally but cannot be returned to a closed pool.
        let retired = self.idle.lock().take();
        drop(retired);
    }
}
