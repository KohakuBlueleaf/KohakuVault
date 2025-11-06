// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Streaming operations implementation for large values
//!
//! This module contains the actual implementation of streaming operations.
//! The main KVault struct in mod.rs provides thin wrappers that delegate to these functions.

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods};
use rusqlite::params;

use crate::common::{open_blob, VaultError};

use super::{_KVault, ops::to_key_bytes};

impl _KVault {
    /// Streamed PUT: read from a Python file-like object (must support read(n)), with known size.
    ///
    /// This is more efficient than regular put() for large values as it avoids
    /// loading the entire value into memory.
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    /// * `reader` - File-like object with read(n) method
    /// * `size` - Total size in bytes
    /// * `chunk_size` - Optional chunk size for reading (default: vault's chunk_size)
    ///
    /// # Example
    /// ```python
    /// with open('large_file.bin', 'rb') as f:
    ///     vault.put_stream(b'my_key', f, size=os.path.getsize('large_file.bin'))
    /// ```
    pub(crate) fn put_stream_impl(
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
            INSERT INTO {t}(key, value)
            VALUES (?1, zeroblob(?2))
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value
            ",
            t = self.table
        );
        let conn = self.conn.lock().unwrap();
        conn.execute(&sql, params![&k, size as i64])
            .map_err(VaultError::from)?;

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
            .map_err(VaultError::from)?;

        // 3) Open BLOB for incremental write
        let mut blob = open_blob(&conn, &self.table, "value", rowid, false)?;

        // 4) Copy in chunks
        let mut written: usize = 0;
        while written < size {
            let to_read = std::cmp::min(chunk, size - written);
            // Call Python reader.read(to_read)
            let data: Vec<u8> = {
                let pybuf = reader.call_method1("read", (to_read,))?;
                if pybuf.is_none() {
                    return Err(VaultError::Py("reader.read() returned None".into()).into());
                }
                if let Ok(b) = pybuf.downcast::<PyBytes>() {
                    b.as_bytes().to_vec()
                } else {
                    return Err(VaultError::Py("reader.read() must return bytes".into()).into());
                }
            };
            if data.is_empty() {
                break;
            }
            blob.write_at(&data, written).map_err(VaultError::from)?;
            written += data.len();
        }
        if written != size {
            return Err(VaultError::Py(format!(
                "short write: wrote {} of {} bytes",
                written, size
            ))
            .into());
        }
        Ok(())
    }

    /// Stream value into a Python file-like object with write(b) method.
    ///
    /// This is more efficient than regular get() for large values as it avoids
    /// loading the entire value into memory.
    ///
    /// # Arguments
    /// * `key` - Key (bytes or str)
    /// * `writer` - File-like object with write(bytes) method
    /// * `chunk_size` - Optional chunk size for reading (default: vault's chunk_size)
    ///
    /// # Returns
    /// Number of bytes written
    ///
    /// # Example
    /// ```python
    /// with open('output.bin', 'wb') as f:
    ///     bytes_written = vault.get_to_file(b'my_key', f)
    /// ```
    pub(crate) fn get_to_file_impl(
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

        // Fetch rowid & size (using LENGTH() to get blob size without reading blob)
        let conn = self.conn.lock().unwrap();
        let (rowid, size): (i64, i64) = conn
            .query_row(
                &format!(
                    "
                    SELECT rowid, LENGTH(value)
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
                    rusqlite::Error::QueryReturnedNoRows => {
                        VaultError::NotFound("Key not found".to_string()).into()
                    }
                    other => VaultError::from(other).into(),
                }
            })?;

        let blob = open_blob(&conn, &self.table, "value", rowid, true)?;

        let mut offset: usize = 0;
        let total = size as usize;
        let mut buf = vec![0u8; chunk];
        while offset < total {
            let to_read = std::cmp::min(chunk, total - offset);
            let n = blob
                .read_at(&mut buf[..to_read], offset)
                .map_err(VaultError::from)?;
            if n == 0 {
                break;
            }
            writer.call_method1("write", (PyBytes::new_bound(py, &buf[..n]),))?;
            offset += n;
        }
        Ok(offset)
    }
}
