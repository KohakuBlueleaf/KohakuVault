//! Pure-Rust KVault client over the KohakuVault SQLite key-value format.
//!
//! Uses the exact same SQLite schema, metadata table, 10-byte encoding headers,
//! write-back cache, and pooled read-only snapshots as the Python `_KVault` binding.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use rusqlite::{
    params, params_from_iter, types::Value, Connection, OptionalExtension, TransactionBehavior,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::cache::{KvCache, SharedValue};
use super::header::{EncodingType, Header, HEADER_SIZE};
use crate::common::meta::{MetaTable, KV_FEATURES_KEY, KV_FEATURE_HEADERS};
use crate::common::read_pool::ReadPool;
use crate::common::{
    checkpoint_wal, open_connection, open_readonly_connection, reader_pool, CacheError, VaultError,
};

/// Options for opening a pure-Rust [`KVault`].
#[derive(Debug, Clone)]
pub struct KVaultOptions {
    pub table: String,
    pub enable_wal: bool,
    pub page_size: u32,
    pub mmap_size: u64,
    pub cache_kb: i64,
    pub read_only: bool,
    pub reader_cache_kb: i64,
    pub reader_idle_limit: usize,
}

impl Default for KVaultOptions {
    fn default() -> Self {
        Self {
            table: "kv".to_string(),
            enable_wal: true,
            page_size: 32768,
            mmap_size: 268_435_456,
            cache_kb: 100_000,
            read_only: false,
            reader_cache_kb: 8192,
            reader_idle_limit: 8,
        }
    }
}

impl KVaultOptions {
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

fn validate_identifier(name: &str) -> Result<(), VaultError> {
    if name.is_empty() || name.chars().any(|c| !c.is_alphanumeric() && c != '_') {
        return Err(VaultError::Invalid(format!(
            "invalid SQL table name '{name}': only ASCII alphanumeric and '_' are allowed"
        )));
    }
    Ok(())
}

fn prefix_end(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(byte) = end.pop() {
        if byte != 255 {
            end.push(byte + 1);
            return Some(end);
        }
    }
    None
}

/// Key/value byte pair returned by [`KVault::scan_items`].
pub type KvEntry = (Vec<u8>, Vec<u8>);

/// Pure-Rust SQLite key-value store compatible with Python `kohakuvault.KVault`.
pub struct KVault {
    conn: parking_lot::Mutex<Option<Connection>>,
    mutation_gate: RwLock<()>,
    closed: AtomicBool,
    readers: Option<ReadPool>,
    read_only: bool,
    table: String,
    cache: Mutex<Option<KvCache>>,
}

impl KVault {
    /// Open or create a KVault with default options (`table = "kv"`).
    pub fn open(path: &str, read_only: bool) -> Result<Self, VaultError> {
        Self::open_with(path, KVaultOptions::default().read_only(read_only))
    }

    /// Open or create a KVault with explicit [`KVaultOptions`].
    pub fn open_with(path: &str, opts: KVaultOptions) -> Result<Self, VaultError> {
        validate_identifier(&opts.table)?;
        if opts.reader_cache_kb <= 0 {
            return Err(VaultError::Invalid("reader_cache_kb must be positive".into()));
        }

        let conn = if opts.read_only {
            open_readonly_connection(path, opts.mmap_size, opts.cache_kb)?
        } else {
            open_connection(path, opts.enable_wal, opts.page_size, opts.mmap_size, opts.cache_kb)?
        };

        if !opts.read_only {
            MetaTable::ensure_table(&conn)?;
            conn.execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {} (
                    key   BLOB PRIMARY KEY NOT NULL,
                    value BLOB NOT NULL
                );",
                opts.table
            ))?;
            let _ = MetaTable::register_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_HEADERS);
        } else {
            conn.prepare(&format!("SELECT key, value FROM {} LIMIT 0", opts.table))?;
        }

        let readers = reader_pool(path).map(|pool| {
            pool.configured(opts.reader_cache_kb, opts.mmap_size, opts.reader_idle_limit)
        });

        Ok(Self {
            conn: parking_lot::Mutex::new(Some(conn)),
            mutation_gate: RwLock::new(()),
            closed: AtomicBool::new(false),
            readers,
            read_only: opts.read_only,
            table: opts.table,
            cache: Mutex::new(None),
        })
    }

    fn ensure_open(&self) -> Result<(), VaultError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(VaultError::Closed);
        }
        Ok(())
    }

    fn ensure_writable(&self) -> Result<(), VaultError> {
        self.ensure_open()?;
        if self.read_only {
            return Err(VaultError::ReadOnly(format!("KVault '{}' is read-only", self.table)));
        }
        Ok(())
    }

    fn with_point_reader<T>(
        &self,
        op: impl FnOnce(&Connection) -> Result<T, VaultError>,
    ) -> Result<T, VaultError> {
        self.ensure_open()?;
        if let Some(pool) = &self.readers {
            return pool.query(op);
        }
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        op(conn)
    }

    fn with_reader<T>(
        &self,
        op: impl FnOnce(&Connection) -> Result<T, VaultError>,
    ) -> Result<T, VaultError> {
        self.ensure_open()?;
        if let Some(pool) = &self.readers {
            return pool.read(op);
        }
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        op(conn)
    }

    fn write_direct(&self, key: &[u8], value: &[u8]) -> Result<(), VaultError> {
        let _mutation = self.mutation_gate.write().unwrap();
        self.ensure_writable()?;
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let sql = format!("INSERT OR REPLACE INTO {} (key, value) VALUES (?1, ?2)", self.table);
        conn.prepare_cached(&sql)?.execute(params![key, value])?;
        if let Some(cache) = self.cache.lock().unwrap().as_mut() {
            cache.remove(key);
        }
        Ok(())
    }

    /// Enable write-back caching with `cap_bytes` capacity and `flush_threshold` auto-flush trigger.
    pub fn enable_cache(&self, cap_bytes: usize, flush_threshold: usize) -> Result<(), VaultError> {
        self.ensure_writable()?;
        if cap_bytes == 0 || flush_threshold == 0 {
            return Err(VaultError::Invalid("cache sizes must be positive".into()));
        }
        let _mutation = self.mutation_gate.write().unwrap();
        let mut guard = self.cache.lock().unwrap();
        if guard.is_none() {
            *guard = Some(KvCache::new(cap_bytes, flush_threshold.min(cap_bytes)));
        }
        Ok(())
    }

    /// Flush pending cached writes atomically to SQLite.
    pub fn flush_cache(&self) -> Result<usize, VaultError> {
        self.ensure_writable()?;
        let mut conn_guard = self.conn.lock();
        let conn = conn_guard.as_mut().ok_or(VaultError::Closed)?;
        let batch = {
            let mut cache_guard = self.cache.lock().unwrap();
            cache_guard.as_mut().and_then(KvCache::begin_flush)
        };
        let Some(batch) = batch else {
            return Ok(0);
        };
        let count = batch.map.len();
        let result = (|| -> Result<(), VaultError> {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let sql = format!("INSERT OR REPLACE INTO {} (key, value) VALUES (?1, ?2)", self.table);
            {
                let mut stmt = tx.prepare_cached(&sql)?;
                for (key, value) in &batch.map {
                    stmt.execute(params![key, value.as_slice()])?;
                }
            }
            tx.commit()?;
            Ok(())
        })();
        drop(batch);
        let retired = self
            .cache
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|cache| cache.finish_flush(result.is_ok()));
        drop(retired);
        result?;
        Ok(count)
    }

    /// Flush and disable the write-back cache.
    pub fn disable_cache(&self) -> Result<usize, VaultError> {
        self.ensure_writable()?;
        let _mutation = self.mutation_gate.write().unwrap();
        let flushed = self.flush_cache()?;
        *self.cache.lock().unwrap() = None;
        Ok(flushed)
    }

    /// Store raw bytes at `key` (no header added; stays raw bytes in Python `KVault` too).
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), VaultError> {
        self.ensure_writable()?;
        let data: SharedValue = Arc::new(value.to_vec());
        loop {
            let mutation = self.mutation_gate.read().unwrap();
            self.ensure_open()?;
            let mut guard = self.cache.lock().unwrap();
            let Some(cache) = guard.as_mut() else {
                drop(guard);
                drop(mutation);
                break;
            };
            match cache.insert(key, &data) {
                Ok(()) => {
                    let flush = cache.should_flush();
                    drop(guard);
                    drop(mutation);
                    if flush {
                        self.flush_cache()?;
                    }
                    return Ok(());
                }
                Err(CacheError::ValueTooLarge) => {
                    drop(guard);
                    drop(mutation);
                    break;
                }
                Err(CacheError::NeedFlush) => {
                    drop(guard);
                    drop(mutation);
                    self.flush_cache()?;
                }
            }
        }
        self.write_direct(key, value)
    }

    /// Atomically commit multiple `(key, value)` pairs in one immediate SQLite transaction.
    pub fn put_many(&self, entries: &[(&[u8], &[u8])]) -> Result<usize, VaultError> {
        self.ensure_writable()?;
        if entries.is_empty() {
            return Ok(0);
        }
        let _mutation = self.mutation_gate.write().unwrap();
        self.ensure_open()?;
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let sql = format!("INSERT OR REPLACE INTO {} (key, value) VALUES (?1, ?2)", self.table);
        {
            let mut stmt = tx.prepare_cached(&sql)?;
            for (k, v) in entries {
                stmt.execute(params![k, v])?;
            }
        }
        tx.commit()?;
        if let Some(cache) = self.cache.lock().unwrap().as_mut() {
            for (k, _) in entries {
                cache.remove(k);
            }
        }
        Ok(entries.len())
    }

    /// Read raw bytes at `key`. Returns `Ok(None)` if absent.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, VaultError> {
        self.ensure_open()?;
        if let Some(cached) = self.cache.lock().unwrap().as_ref().and_then(|c| c.get(key)) {
            return Ok(Some((*cached).clone()));
        }
        self.with_point_reader(|conn| {
            let sql = format!("SELECT value FROM {} WHERE key = ?1", self.table);
            Ok(conn
                .prepare_cached(&sql)?
                .query_row([key], |row| row.get(0))
                .optional()?)
        })
    }

    /// Read multiple keys in input order using one consistent snapshot.
    pub fn get_many(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, VaultError> {
        self.ensure_open()?;
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(keys.len());
        let mut misses: Vec<(usize, &[u8])> = Vec::new();
        {
            let guard = self.cache.lock().unwrap();
            for (idx, key) in keys.iter().enumerate() {
                if let Some(cached) = guard.as_ref().and_then(|c| c.get(key)) {
                    results.push(Some((*cached).clone()));
                } else {
                    results.push(None);
                    misses.push((idx, *key));
                }
            }
        }
        if misses.is_empty() {
            return Ok(results);
        }
        self.with_reader(|conn| {
            let sql = format!("SELECT value FROM {} WHERE key = ?1", self.table);
            let mut stmt = conn.prepare_cached(&sql)?;
            for (idx, key) in misses {
                results[idx] = stmt.query_row([key], |row| row.get(0)).optional()?;
            }
            Ok(())
        })?;
        Ok(results)
    }

    /// Serialize `value` as MessagePack with KohakuVault's standard 10-byte header so both
    /// Rust and Python `KVault` decode it identically.
    pub fn put_msgpack<T: Serialize>(&self, key: &[u8], value: &T) -> Result<(), VaultError> {
        let payload =
            rmp_serde::to_vec_named(value).map_err(|e| VaultError::Codec(e.to_string()))?;
        let header = Header::new(EncodingType::MessagePack).encode();
        let mut buf = Vec::with_capacity(HEADER_SIZE + payload.len());
        buf.extend_from_slice(&header);
        buf.extend_from_slice(&payload);
        self.put(key, &buf)
    }

    /// Read and deserialize a MessagePack value (strips KohakuVault's 10-byte header if present).
    pub fn get_msgpack<T: DeserializeOwned>(&self, key: &[u8]) -> Result<Option<T>, VaultError> {
        let Some(raw) = self.get(key)? else {
            return Ok(None);
        };
        let slice = match Header::decode(&raw).map_err(VaultError::Codec)? {
            Some(hdr) if hdr.encoding == EncodingType::MessagePack => &raw[HEADER_SIZE..],
            Some(hdr) if hdr.encoding == EncodingType::Json => {
                let parsed = serde_json::from_slice(&raw[HEADER_SIZE..])
                    .map_err(|e| VaultError::Codec(e.to_string()))?;
                return Ok(Some(parsed));
            }
            Some(hdr) => {
                return Err(VaultError::Codec(format!(
                    "expected MessagePack or JSON encoding, found {:?}",
                    hdr.encoding
                )));
            }
            None => raw.as_slice(),
        };
        let parsed = rmp_serde::from_slice(slice).map_err(|e| VaultError::Codec(e.to_string()))?;
        Ok(Some(parsed))
    }

    /// Check if `key` exists.
    pub fn exists(&self, key: &[u8]) -> Result<bool, VaultError> {
        self.ensure_open()?;
        if self
            .cache
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|c| c.contains_key(key))
        {
            return Ok(true);
        }
        self.with_point_reader(|conn| {
            let sql = format!("SELECT 1 FROM {} WHERE key = ?1", self.table);
            Ok(conn
                .prepare_cached(&sql)?
                .query_row([key], |_| Ok(()))
                .optional()?
                .is_some())
        })
    }

    /// Delete `key`. Returns `true` if the key existed.
    pub fn delete(&self, key: &[u8]) -> Result<bool, VaultError> {
        let _mutation = self.mutation_gate.write().unwrap();
        self.ensure_writable()?;
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let sql = format!("DELETE FROM {} WHERE key = ?1", self.table);
        let deleted = conn.prepare_cached(&sql)?.execute([key])?;
        let pending = self
            .cache
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|c| c.remove(key));
        Ok(deleted > 0 || pending.is_some())
    }

    /// Delete all entries in this table atomically.
    pub fn clear(&self) -> Result<(), VaultError> {
        let _mutation = self.mutation_gate.write().unwrap();
        self.ensure_writable()?;
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        conn.execute(&format!("DELETE FROM {}", self.table), [])?;
        if let Some(cache) = self.cache.lock().unwrap().as_mut() {
            let _ = cache.take_pending();
        }
        Ok(())
    }

    /// Count keys (committed + pending overlay).
    pub fn count(&self) -> Result<usize, VaultError> {
        self.ensure_open()?;
        let has_cache = self
            .cache
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|c| !c.is_empty());
        if !has_cache {
            return self.with_point_reader(|conn| {
                let sql = format!("SELECT COUNT(*) FROM {}", self.table);
                Ok(conn.query_row(&sql, [], |r| r.get::<_, i64>(0))? as usize)
            });
        }
        Ok(self.scan_keys(None, None, None)?.len())
    }

    /// Keyset-paginated key scan.
    pub fn scan_keys(
        &self,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
        after: Option<&[u8]>,
    ) -> Result<Vec<Vec<u8>>, VaultError> {
        Ok(self
            .scan_page(prefix, limit, after, false)?
            .into_iter()
            .map(|(k, _)| k)
            .collect())
    }

    /// Keyset-paginated `(key, value)` scan.
    pub fn scan_items(
        &self,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
        after: Option<&[u8]>,
    ) -> Result<Vec<KvEntry>, VaultError> {
        Ok(self
            .scan_page(prefix, limit, after, true)?
            .into_iter()
            .map(|(k, v)| (k, (*v).clone()))
            .collect())
    }

    fn scan_page(
        &self,
        prefix: Option<&[u8]>,
        limit: Option<usize>,
        after: Option<&[u8]>,
        include_values: bool,
    ) -> Result<Vec<(Vec<u8>, SharedValue)>, VaultError> {
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let sql_limit = limit
            .map(i64::try_from)
            .transpose()
            .map_err(|_| VaultError::Invalid("iteration limit is too large".into()))?
            .unwrap_or(-1);
        let fields = if include_values { "key, value" } else { "key" };
        let mut sql = format!("SELECT {fields} FROM {} WHERE 1 = 1", self.table);
        let mut parameters = Vec::<Value>::new();

        if let Some(p) = prefix {
            sql.push_str(" AND key >= ?");
            parameters.push(Value::Blob(p.to_vec()));
            if let Some(end) = prefix_end(p) {
                sql.push_str(" AND key < ?");
                parameters.push(Value::Blob(end));
            }
        }
        if let Some(a) = after {
            sql.push_str(" AND key > ?");
            parameters.push(Value::Blob(a.to_vec()));
        }
        sql.push_str(" ORDER BY key LIMIT ?");
        parameters.push(Value::Integer(sql_limit));

        let pending = {
            let guard = self.cache.lock().unwrap();
            guard
                .as_ref()
                .map(|c| c.page(prefix, after, limit, include_values))
                .unwrap_or_default()
        };

        let empty: SharedValue = Arc::new(Vec::new());
        let rows: Vec<(Vec<u8>, SharedValue)> = self.with_reader(|conn| {
            let mut stmt = conn.prepare_cached(&sql)?;
            let mapped = stmt.query_map(params_from_iter(parameters), |row| {
                let val = if include_values {
                    Arc::new(row.get::<_, Vec<u8>>(1)?)
                } else {
                    Arc::clone(&empty)
                };
                Ok((row.get::<_, Vec<u8>>(0)?, val))
            })?;
            Ok(mapped.collect::<Result<Vec<_>, _>>()?)
        })?;

        if pending.is_empty() {
            return Ok(rows);
        }
        let mut merged: BTreeMap<_, _> = rows.into_iter().collect();
        merged.extend(pending);
        Ok(merged
            .into_iter()
            .take(limit.unwrap_or(usize::MAX))
            .collect())
    }

    /// Passively checkpoint the SQLite WAL.
    pub fn checkpoint(&self) -> Result<i64, VaultError> {
        self.ensure_writable()?;
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        checkpoint_wal(conn)
    }

    /// Flush any pending cache writes and close connections.
    pub fn close(&self) -> Result<(), VaultError> {
        let _mutation = self.mutation_gate.write().unwrap();
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        if !self.read_only {
            self.flush_cache()?;
        }
        self.closed.store(true, Ordering::Release);
        if let Some(pool) = &self.readers {
            pool.close();
        }
        let _ = self.conn.lock().take();
        *self.cache.lock().unwrap() = None;
        Ok(())
    }
}

impl Drop for KVault {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::{KVault, KVaultOptions};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct SampleMeta {
        name: String,
        step: i64,
    }

    #[test]
    fn kv_crud_batch_scan_and_msgpack() {
        let kv =
            KVault::open_with(":memory:", KVaultOptions::default().with_table("meta")).unwrap();
        kv.enable_cache(1 << 20, 512 << 10).unwrap();
        kv.put(b"a/1", b"one").unwrap();
        kv.put(b"a/2", b"two").unwrap();
        assert_eq!(kv.get(b"a/1").unwrap().as_deref(), Some(b"one".as_slice()));
        assert_eq!(kv.flush_cache().unwrap(), 2);

        kv.put_many(&[(b"b/1", b"x"), (b"b/2", b"y")]).unwrap();
        let got = kv.get_many(&[b"b/2", b"missing", b"a/1"]).unwrap();
        assert_eq!(got, vec![Some(b"y".to_vec()), None, Some(b"one".to_vec())]);

        let keys = kv.scan_keys(Some(b"a/"), None, None).unwrap();
        assert_eq!(keys, vec![b"a/1".to_vec(), b"a/2".to_vec()]);

        let meta = SampleMeta { name: "run-1".into(), step: 42 };
        kv.put_msgpack(b"run/info", &meta).unwrap();
        assert_eq!(kv.get_msgpack::<SampleMeta>(b"run/info").unwrap(), Some(meta));
        assert!(kv.exists(b"run/info").unwrap());
        assert!(kv.delete(b"a/1").unwrap());
        assert!(!kv.exists(b"a/1").unwrap());
        kv.close().unwrap();
    }
}
