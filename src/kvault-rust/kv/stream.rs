//! Transactional streaming with validated Python I/O callbacks.

use std::cell::RefCell;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes};
use rusqlite::{params, TransactionBehavior};

use crate::common::{open_blob, VaultError};

use super::{_KVault, ops::to_key_bytes};

thread_local! {
    static CALLBACK_VAULTS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct StreamError(PyErr);

impl From<rusqlite::Error> for StreamError {
    fn from(error: rusqlite::Error) -> Self {
        Self(VaultError::from(error).into())
    }
}

struct CallbackGuard;

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        CALLBACK_VAULTS.with(|active| {
            active.borrow_mut().pop();
        });
    }
}

impl _KVault {
    pub(crate) fn check_stream_reentry(&self) -> Result<(), VaultError> {
        let address = self as *const Self as usize;
        if CALLBACK_VAULTS.with(|active| active.borrow().contains(&address)) {
            return Err(VaultError::Py(
                "reentrant access from a stream callback is not supported".into(),
            ));
        }
        Ok(())
    }

    fn stream_callback<T>(&self, callback: impl FnOnce(Python<'_>) -> PyResult<T>) -> PyResult<T> {
        CALLBACK_VAULTS.with(|active| active.borrow_mut().push(self as *const Self as usize));
        let _guard = CallbackGuard;
        Python::with_gil(callback)
    }

    fn stream_chunk(&self, requested: Option<usize>) -> PyResult<usize> {
        let chunk = requested.unwrap_or(self.chunk_size);
        if chunk == 0 || chunk > i32::MAX as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be between 1 and 2147483647",
            ));
        }
        Ok(chunk)
    }

    fn write_stream_chunk(&self, writer: &Py<PyAny>, bytes: &[u8]) -> PyResult<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            let count = self.stream_callback(|py| {
                let result = writer
                    .bind(py)
                    .call_method1("write", (PyBytes::new(py, &bytes[offset..]),))?;
                if result.is_none() || result.is_instance_of::<PyBool>() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "writer.write() must return a positive byte count",
                    ));
                }
                let count: usize = result.extract()?;
                if count == 0 || count > bytes.len() - offset {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "writer.write() returned an invalid byte count",
                    ));
                }
                Ok(count)
            })?;
            offset += count;
        }
        Ok(())
    }

    pub(crate) fn put_stream_impl(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        reader: &Bound<'_, PyAny>,
        size: usize,
        chunk_size: Option<usize>,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        let key = to_key_bytes(py, key)?;
        let chunk = self.stream_chunk(chunk_size)?;
        if size > i32::MAX as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stream size exceeds SQLite BLOB limit",
            ));
        }
        let reader = reader.clone().unbind();
        py.allow_threads(move || {
            let _mutation = self.mutation_gate.write().unwrap();
            self.ensure_open()?;
            let mut conn = self.connection()?;
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(VaultError::from)?;
            tx.execute(
                &format!("INSERT INTO {}(key, value) VALUES (?1, zeroblob(?2)) ON CONFLICT(key) DO UPDATE SET value=excluded.value", self.table),
                params![&key, size as i64],
            ).map_err(VaultError::from)?;

            if size > 0 {
                let rowid: i64 = tx.query_row(
                    &format!("SELECT rowid FROM {} WHERE key=?1", self.table),
                    [&key], |row| row.get(0),
                ).map_err(VaultError::from)?;
                let mut blob = open_blob(&tx, &self.table, "value", rowid, false)?;
                let mut written = 0;
                while written < size {
                    let requested = chunk.min(size - written);
                    let data = self.stream_callback(|py| {
                        let result = reader.bind(py).call_method1("read", (requested,))?;
                        let bytes = result.downcast::<PyBytes>().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err("reader.read() must return bytes")
                        })?;
                        if bytes.as_bytes().len() > requested {
                            return Err(pyo3::exceptions::PyValueError::new_err("reader.read() returned more bytes than requested"));
                        }
                        Ok(bytes.as_bytes().to_vec())
                    })?;
                    if data.is_empty() {
                        return Err(VaultError::Py(format!("short write: wrote {written} of {size} bytes")).into());
                    }
                    blob.write_at(&data, written).map_err(VaultError::from)?;
                    written += data.len();
                }
            }
            // BLOB handles are closed before publication; failed transactions retain pending values.
            tx.commit().map_err(VaultError::from)?;
            self.remove_pending(&key);
            Ok(())
        })
    }

    pub(crate) fn get_to_file_impl(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        writer: &Bound<'_, PyAny>,
        chunk_size: Option<usize>,
    ) -> PyResult<usize> {
        self.ensure_open()?;
        let key = to_key_bytes(py, key)?;
        let chunk = self.stream_chunk(chunk_size)?;
        let writer = writer.clone().unbind();
        py.allow_threads(move || {
            let cached = self
                .cache
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|cache| cache.get(&key));
            if let Some(value) = cached {
                for bytes in value.chunks(chunk) {
                    self.write_stream_chunk(&writer, bytes)?;
                }
                return Ok(value.len());
            }
            let read = |conn: &rusqlite::Connection| -> PyResult<usize> {
                let (rowid, size): (i64, usize) = conn
                    .query_row(
                        &format!("SELECT rowid, LENGTH(value) FROM {} WHERE key=?1", self.table),
                        [&key],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .map_err(|error| match error {
                        rusqlite::Error::QueryReturnedNoRows => {
                            VaultError::NotFound("Key not found".into())
                        }
                        other => VaultError::from(other),
                    })?;
                if size > 0 {
                    let blob = open_blob(conn, &self.table, "value", rowid, true)?;
                    let mut buffer = vec![0; chunk.min(size)];
                    let mut offset = 0;
                    while offset < size {
                        let count = buffer.len().min(size - offset);
                        let read = blob
                            .read_at(&mut buffer[..count], offset)
                            .map_err(VaultError::from)?;
                        if read == 0 {
                            return Err(
                                VaultError::Py("unexpected end of stored BLOB".into()).into()
                            );
                        }
                        self.write_stream_chunk(&writer, &buffer[..read])?;
                        offset += read;
                    }
                }
                Ok(size)
            };
            if let Some(readers) = &self.readers {
                readers
                    .read::<_, StreamError>(|conn| read(conn).map_err(StreamError))
                    .map_err(|error| error.0)
            } else {
                let conn = self.connection()?;
                let tx = conn.unchecked_transaction().map_err(VaultError::from)?;
                let result = read(&tx);
                tx.rollback().map_err(VaultError::from)?;
                result
            }
        })
    }
}
