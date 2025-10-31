use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyString, PyStringMethods};
use rusqlite::blob::Blob;
use rusqlite::{params, Connection, OpenFlags};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
enum KvError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Python error: {0}")]
    Py(String),
    #[error("Key not found")]
    NotFound,
}

impl From<KvError> for PyErr {
    fn from(e: KvError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

fn to_key_bytes(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<Vec<u8>, KvError> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        Ok(b.as_bytes().to_vec())
    } else if let Ok(s) = obj.downcast::<PyString>() {
        Ok(s.to_str()
            .map_err(|e| KvError::Py(format!("Invalid UTF-8: {}", e)))?
            .as_bytes()
            .to_vec())
    } else {
        Err(KvError::Py("key must be bytes or str".into()))
    }
}

struct WriteBackCache {
    map: HashMap<Vec<u8>, Vec<u8>>,
    current_bytes: usize,
    cap_bytes: usize,
    flush_threshold: usize,
}

impl WriteBackCache {
    fn new(cap_bytes: usize, flush_threshold: usize) -> Self {
        Self { map: HashMap::new(), current_bytes: 0, cap_bytes, flush_threshold }
    }
    fn insert(&mut self, k: Vec<u8>, v: Vec<u8>) {
        let add = k.len() + v.len();
        self.current_bytes += add;
        self.map.insert(k, v);
    }
    fn should_flush(&self) -> bool {
        self.current_bytes >= self.flush_threshold || self.current_bytes >= self.cap_bytes
    }
    fn drain(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::with_capacity(self.map.len());
        for (k, v) in self.map.drain() {
            out.push((k, v));
        }
        self.current_bytes = 0;
        out
    }
}

#[pyclass]
struct _KVault {
    conn: Mutex<Connection>,
    table: String,
    cache: Mutex<Option<WriteBackCache>>,
    chunk_size: usize,
}

#[pymethods]
impl _KVault {
    /// path: SQLite file path
    /// table: table name (default "kv")
    /// chunk_size: streaming chunk size in bytes (default 1 MiB)
    /// enable_wal, page_size, mmap_size, cache_pages_kb are tunables
    #[new]
    #[pyo3(signature = (path, table="kv", chunk_size=1<<20, enable_wal=true, page_size=32768, mmap_size=268_435_456, cache_kb=100_000))]
    fn new(
        _py: Python<'_>,
        path: &str,
        table: &str,
        chunk_size: usize,
        enable_wal: bool,
        page_size: u32,
        mmap_size: u64,
        cache_kb: i64,
    ) -> PyResult<Self> {
        let is_new = !Path::new(path).exists();
        if let Some(dir) = Path::new(path).parent() {
            fs::create_dir_all(dir).ok();
        }

        // Open read-write, create if missing.
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_URI;

        let conn = Connection::open_with_flags(path, flags).map_err(KvError::from)?;

        // Tuning pragmas
        if is_new {
            // page_size only sticks on empty DB
            let _ = conn.execute_batch(&format!("PRAGMA page_size={};", page_size));
        }
        if enable_wal {
            let _ = conn.pragma_update(None, "journal_mode", &"WAL");
        }
        let _ = conn.pragma_update(None, "synchronous", &"NORMAL");
        let _ = conn.pragma_update(None, "mmap_size", &mmap_size);
        // negative cache_size means KB units
        let _ = conn.pragma_update(None, "cache_size", &(-cache_kb));
        let _ = conn.pragma_update(None, "temp_store", &"MEMORY");
        let _ = conn.busy_timeout(std::time::Duration::from_millis(5000));

        // Schema
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS {t} (
                id    INTEGER PRIMARY KEY,
                key   BLOB    NOT NULL UNIQUE,
                value BLOB    NOT NULL,
                size  INTEGER NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS {t}_key_idx
            ON {t}(key);
            ",
            t = rusqlite::types::ValueRef::from(table)
                .as_str()
                .unwrap_or("kv") // defensive
        ))
        .map_err(KvError::from)?;

        Ok(Self {
            conn: Mutex::new(conn),
            table: table.to_string(),
            cache: Mutex::new(None),
            chunk_size,
        })
    }

    /// Enable/disable write-back cache (bytes-bounded). Flush when threshold reached.
    #[pyo3(signature = (cap_bytes=64<<20, flush_threshold=16<<20))]
    fn enable_cache(&self, cap_bytes: usize, flush_threshold: usize) {
        let mut guard = self.cache.lock().unwrap();
        *guard = Some(WriteBackCache::new(cap_bytes, flush_threshold));
    }
    fn disable_cache(&self) {
        let mut guard = self.cache.lock().unwrap();
        *guard = None;
    }

    /// Insert/replace whole value (bytes-like). Uses UPSERT and sets size.
    fn put(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let k = to_key_bytes(py, key)?;
        let v = if let Ok(b) = value.downcast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else {
            return Err(KvError::Py("value must be bytes-like".into()).into());
        };

        if let Some(cache) = self.cache.lock().unwrap().as_mut() {
            cache.insert(k, v);
            if cache.should_flush() {
                let _ = py; // no-op, just reminding ourselves
                self.flush_cache(py)?;
            }
            return Ok(());
        }

        let size = v.len() as i64;
        let sql = format!(
            "
            INSERT INTO {t}(key, value, size)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                size = excluded.size
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        conn.execute(&sql, params![k, v, size])
            .map_err(KvError::from)?;
        Ok(())
    }

    /// Streamed PUT: read from a Python file-like object (must support read(n)), with known size.
    #[pyo3(signature = (key, reader, size, chunk_size=None))]
    fn put_stream(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        reader: &Bound<'_, PyAny>,
        size: usize,
        chunk_size: Option<usize>,
    ) -> PyResult<()> {
        let k = to_key_bytes(py, key)?;
        let chunk = chunk_size.unwrap_or(self.chunk_size);

        // 1) Upsert a zeroblob of desired size
        let sql = format!(
            "
            INSERT INTO {t}(key, value, size)
            VALUES (?1, zeroblob(?2), ?2)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                size = excluded.size
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        conn.execute(&sql, params![&k, size as i64])
            .map_err(KvError::from)?;

        // 2) Get rowid
        let rowid: i64 = conn
            .query_row(
                &format!(
                    "
                    SELECT rowid
                    FROM {}
                    WHERE key = ?1
                    ",
                    self.table
                ),
                params![&k],
                |r| r.get(0),
            )
            .map_err(KvError::from)?;

        // 3) Open BLOB for incremental write
        let mut blob: Blob = conn
            .blob_open(rusqlite::DatabaseName::Main, &self.table, "value", rowid, true)
            .map_err(KvError::from)?;

        // 4) Copy in chunks
        let mut written: usize = 0;
        while written < size {
            let to_read = std::cmp::min(chunk, size - written);
            // Call Python reader.read(to_read)
            let data: Vec<u8> = {
                let pybuf = reader.call_method1("read", (to_read,))?;
                if pybuf.is_none() {
                    return Err(KvError::Py("reader.read() returned None".into()).into());
                }
                if let Ok(b) = pybuf.downcast::<PyBytes>() {
                    b.as_bytes().to_vec()
                } else {
                    return Err(KvError::Py("reader.read() must return bytes".into()).into());
                }
            };
            if data.is_empty() {
                break;
            }
            blob.write_at(&data, written).map_err(KvError::from)?;
            written += data.len();
        }
        if written != size {
            return Err(
                KvError::Py(format!("short write: wrote {} of {} bytes", written, size)).into()
            );
        }
        Ok(())
    }

    /// Get entire value as bytes (avoid for huge blobs; prefer get_to_file()).
    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        let k = to_key_bytes(py, key)?;

        // Check write-back cache first
        if let Some(cache) = self.cache.lock().unwrap().as_ref() {
            if let Some(v) = cache.map.get(&k) {
                return Ok(PyBytes::new_bound(py, v).unbind());
            }
        }

        let sql = format!(
            "
            SELECT value
            FROM {}
            WHERE key = ?1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let data: Vec<u8> =
            conn.query_row(&sql, params![k], |r| r.get(0))
                .map_err(|e| -> PyErr {
                    match e {
                        rusqlite::Error::QueryReturnedNoRows => KvError::NotFound.into(),
                        other => KvError::from(other).into(),
                    }
                })?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Stream value into a Python file-like object with write(b) method.
    #[pyo3(signature = (key, writer, chunk_size=None))]
    fn get_to_file(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        writer: &Bound<'_, PyAny>,
        chunk_size: Option<usize>,
    ) -> PyResult<usize> {
        let k = to_key_bytes(py, key)?;
        let chunk = chunk_size.unwrap_or(self.chunk_size);

        // Check cache first
        if let Some(cache) = self.cache.lock().unwrap().as_ref() {
            if let Some(v) = cache.map.get(&k) {
                writer.call_method1("write", (PyBytes::new_bound(py, v),))?;
                return Ok(v.len());
            }
        }

        // Fetch rowid & size
        let conn = self.conn.lock().unwrap();
        let (rowid, size): (i64, i64) = conn
            .query_row(
                &format!(
                    "
                    SELECT rowid, size
                    FROM {}
                    WHERE key = ?1
                    ",
                    self.table
                ),
                params![&k],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| -> PyErr {
                match e {
                    rusqlite::Error::QueryReturnedNoRows => KvError::NotFound.into(),
                    other => KvError::from(other).into(),
                }
            })?;

        let blob: Blob = conn
            .blob_open(rusqlite::DatabaseName::Main, &self.table, "value", rowid, false)
            .map_err(KvError::from)?;

        let mut offset: usize = 0;
        let total = size as usize;
        let mut buf = vec![0u8; chunk];
        while offset < total {
            let to_read = std::cmp::min(chunk, total - offset);
            let n = blob
                .read_at(&mut buf[..to_read], offset)
                .map_err(KvError::from)?;
            if n == 0 {
                break;
            }
            writer.call_method1("write", (PyBytes::new_bound(py, &buf[..n]),))?;
            offset += n;
        }
        Ok(offset)
    }

    fn delete(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = to_key_bytes(py, key)?;
        let sql = format!(
            "
            DELETE FROM {}
            WHERE key = ?1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(&sql, params![k]).map_err(KvError::from)?;
        Ok(n > 0)
    }

    fn exists(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = to_key_bytes(py, key)?;
        let sql = format!(
            "
            SELECT 1
            FROM {}
            WHERE key = ?1
            LIMIT 1
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let found: Result<i32, _> = conn.query_row(&sql, params![k], |r| r.get(0));
        Ok(found.is_ok())
    }

    /// Scan keys (optionally prefix for TEXT-ish keys; for binary prefixes, pass bytes).
    #[pyo3(signature = (prefix=None, limit=1000))]
    fn scan_keys(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: usize,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let mut out = Vec::new();
        let conn = self.conn.lock().unwrap();

        if let Some(p) = prefix {
            let k = to_key_bytes(py, p)?;
            // Simple prefix scan with range [prefix, prefix||0xFF...].
            // Works for raw bytes because SQLite compares blobs lexicographically.
            let mut hi = k.clone();
            hi.push(0xFF);
            let sql = format!(
                "
                SELECT key
                FROM {}
                WHERE key >= ?1 AND key < ?2
                ORDER BY key
                LIMIT ?3
                ",
                self.table
            );
            let mut stmt = conn.prepare(&sql).map_err(KvError::from)?;
            let iter = stmt
                .query_map(params![k, hi, limit as i64], |r| r.get::<_, Vec<u8>>(0))
                .map_err(KvError::from)?;
            for r in iter {
                let kb = r.map_err(KvError::from)?;
                out.push(PyBytes::new_bound(py, &kb).unbind());
            }
        } else {
            let sql = format!(
                "
                SELECT key
                FROM {}
                ORDER BY key
                LIMIT ?1
                ",
                self.table
            );
            let mut stmt = conn.prepare(&sql).map_err(KvError::from)?;
            let iter = stmt
                .query_map(params![limit as i64], |r| r.get::<_, Vec<u8>>(0))
                .map_err(KvError::from)?;
            for r in iter {
                let kb = r.map_err(KvError::from)?;
                out.push(PyBytes::new_bound(py, &kb).unbind());
            }
        }
        Ok(out)
    }

    /// Flush write-back cache (if enabled) in a single transaction.
    fn flush_cache(&self, _py: Python<'_>) -> PyResult<usize> {
        let mut guard = self.cache.lock().unwrap();
        let Some(cache) = guard.as_mut() else {
            return Ok(0);
        };
        let entries = cache.drain();
        if entries.is_empty() {
            return Ok(0);
        }
        let sql = format!(
            "
            INSERT INTO {t}(key, value, size)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                size = excluded.size
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(KvError::from)?;
        let mut stmt = tx.prepare(&sql).map_err(KvError::from)?;
        let mut count = 0usize;
        for (k, v) in entries {
            stmt.execute(params![k, &v, (v.len() as i64)])
                .map_err(KvError::from)?;
            count += 1;
        }
        drop(stmt);
        tx.commit().map_err(KvError::from)?;
        Ok(count)
    }

    /// Vacuum & optimize (blocks writer).
    fn optimize(&self, _py: Python<'_>) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("PRAGMA optimize; VACUUM;")
            .map_err(KvError::from)?;
        Ok(())
    }

    fn len(&self, _py: Python<'_>) -> PyResult<i64> {
        let sql = format!(
            "
            SELECT COUNT(*)
            FROM {}
            ",
            self.table
        );
        let conn = self.conn.lock().unwrap();
        let n: i64 = conn
            .query_row(&sql, [], |r| r.get(0))
            .map_err(KvError::from)?;
        Ok(n)
    }
}

#[pymodule]
fn _kvault(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_KVault>()?;
    Ok(())
}
