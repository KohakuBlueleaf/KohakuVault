//! Python bindings for the buffer-oriented SQLite column engine.

mod chunks;
mod core;
mod fixed;
mod varsize_ops;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::params;
use thiserror::Error;

use core::{meta, ColumnInfo, Core};

type ColResult<T> = Result<T, ColError>;
type VariableBuffers = (Py<PyBytes>, Py<PyBytes>);

#[derive(Debug, Error)]
pub(crate) enum ColError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("column index out of range")]
    Bounds,
    #[error("column not found")]
    NotFound,
    #[error("column vault is closed")]
    Closed,
    #[error("column vault is read-only")]
    ReadOnly,
    #[error("invalid column storage: {0}")]
    Corrupt(String),
}

impl From<ColError> for PyErr {
    fn from(error: ColError) -> Self {
        match error {
            ColError::Invalid(_) | ColError::Closed => PyValueError::new_err(error.to_string()),
            ColError::Bounds => PyIndexError::new_err(error.to_string()),
            ColError::NotFound => PyKeyError::new_err(error.to_string()),
            ColError::ReadOnly => PyPermissionError::new_err(error.to_string()),
            _ => PyRuntimeError::new_err(error.to_string()),
        }
    }
}

#[pyclass]
pub struct _ColumnVault {
    core: Core,
}

#[pymethods]
impl _ColumnVault {
    #[new]
    #[pyo3(signature = (path, read_only=false))]
    fn new(py: Python<'_>, path: &str, read_only: bool) -> PyResult<Self> {
        let core = py.allow_threads(|| Core::open(path, read_only))?;
        Ok(Self { core })
    }

    fn create(
        &self,
        py: Python<'_>,
        name: &str,
        dtype: &str,
        elem_size: usize,
        block_bytes: usize,
    ) -> PyResult<i64> {
        Ok(py.allow_threads(|| self.core.create(name, dtype, elem_size, block_bytes))?)
    }

    fn info(&self, py: Python<'_>, name: &str) -> PyResult<ColumnInfo> {
        Ok(py.allow_threads(|| self.core.info(name))?)
    }

    fn length(&self, py: Python<'_>, id: i64) -> PyResult<usize> {
        Ok(py.allow_threads(|| self.core.read(|conn| Ok(meta(conn, id)?.length)))?)
    }

    fn list_columns(&self, py: Python<'_>) -> PyResult<Vec<(String, String, usize)>> {
        Ok(py.allow_threads(|| {
            self.core.read(|conn| {
                let mut stmt = conn.prepare_cached(
                    "SELECT name, dtype, length FROM kvault_col_columns ORDER BY col_id",
                )?;
                let rows = stmt
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? as usize)))?;
                Ok(rows.collect::<Result<Vec<_>, _>>()?)
            })
        })?)
    }

    fn delete_column(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        Ok(py.allow_threads(|| {
            self.core.write(|tx| {
                Ok(tx.execute("DELETE FROM kvault_col_columns WHERE name = ?1", [name])? > 0)
            })
        })?)
    }

    fn clear(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        Ok(py.allow_threads(|| self.core.clear(id))?)
    }

    fn close(&self, py: Python<'_>) {
        py.allow_threads(|| self.core.close());
    }

    fn checkpoint_wal(&self, py: Python<'_>) -> PyResult<(i64, i64, i64)> {
        Ok(py.allow_threads(|| self.core.checkpoint())?)
    }

    fn append_fixed(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        // Python bytes are immutable and stay alive while the GIL is released.
        let input = data.as_bytes();
        Ok(py.allow_threads(|| {
            self.core
                .write(|tx| fixed::append(tx, &meta(tx, id)?, input))
        })?)
    }

    #[pyo3(signature = (id, start=0, count=None))]
    fn read_fixed(
        &self,
        py: Python<'_>,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let output = py.allow_threads(|| {
            self.core
                .read(|conn| fixed::read(conn, &meta(conn, id)?, start, count))
        })?;
        Ok(PyBytes::new(py, &output).unbind())
    }

    fn take_fixed(&self, py: Python<'_>, id: i64, indices: Vec<i64>) -> PyResult<Py<PyBytes>> {
        let output = py.allow_threads(|| {
            self.core
                .read(|conn| fixed::take(conn, &meta(conn, id)?, &indices))
        })?;
        Ok(PyBytes::new(py, &output).unbind())
    }

    fn overwrite_fixed(
        &self,
        py: Python<'_>,
        id: i64,
        start: usize,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        let input = data.as_bytes();
        Ok(py.allow_threads(|| {
            self.core
                .write(|tx| fixed::overwrite(tx, &meta(tx, id)?, start, input))
        })?)
    }

    fn append_variable(
        &self,
        py: Python<'_>,
        id: i64,
        data: &Bound<'_, PyBytes>,
        offsets: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        let input = data.as_bytes();
        let encoded = offsets.as_bytes();
        Ok(py.allow_threads(|| {
            let positions = varsize_ops::decode_offsets(encoded, input.len())?;
            self.core
                .write(|tx| varsize_ops::append(tx, &meta(tx, id)?, input, &positions))
        })?)
    }

    #[pyo3(signature = (id, start=0, count=None))]
    fn read_variable(
        &self,
        py: Python<'_>,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<VariableBuffers> {
        let (payload, offsets) = py.allow_threads(|| {
            self.core
                .read(|conn| varsize_ops::read(conn, &meta(conn, id)?, start, count))
        })?;
        Ok((PyBytes::new(py, &payload).unbind(), PyBytes::new(py, &offsets).unbind()))
    }

    fn replace_fixed(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let input = data.as_bytes();
        Ok(py.allow_threads(|| {
            self.core.write(|tx| {
                let mut metadata = meta(tx, id)?;
                tx.execute("DELETE FROM kvault_col_blocks WHERE col_id = ?1", [id])?;
                metadata.length = 0;
                fixed::append(tx, &metadata, input)
            })
        })?)
    }

    fn replace_variable(
        &self,
        py: Python<'_>,
        id: i64,
        data: &Bound<'_, PyBytes>,
        offsets: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        let input = data.as_bytes();
        let encoded = offsets.as_bytes();
        Ok(py.allow_threads(|| {
            let positions = varsize_ops::decode_offsets(encoded, input.len())?;
            self.core.write(|tx| {
                let mut metadata = meta(tx, id)?;
                tx.execute("DELETE FROM kvault_col_blocks WHERE col_id = ?1", [id])?;
                metadata.length = 0;
                varsize_ops::append(tx, &metadata, input, &positions)
            })
        })?)
    }

    fn storage_info(&self, py: Python<'_>, id: i64) -> PyResult<(usize, usize, usize)> {
        Ok(py.allow_threads(|| {
            self.core.read(|conn| {
                meta(conn, id)?;
                Ok(conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(used_bytes), 0), COALESCE(SUM(capacity_bytes), 0)
                 FROM kvault_col_blocks WHERE col_id = ?1", params![id],
                |row| Ok((row.get::<_, i64>(0)? as usize, row.get::<_, i64>(1)? as usize,
                    row.get::<_, i64>(2)? as usize)),
            )?)
            })
        })?)
    }
}
