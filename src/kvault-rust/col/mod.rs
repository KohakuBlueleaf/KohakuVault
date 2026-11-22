//! Buffer-oriented SQLite column engine with optional Python bindings.

mod chunks;
mod core;
mod fixed;
mod varsize_ops;

#[cfg(feature = "python")]
use pyo3::exceptions::{PyIndexError, PyKeyError, PyPermissionError, PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;
use rusqlite::params;
use thiserror::Error;

pub use core::ColumnInfo;
use core::{meta, Core};

pub type ColResult<T> = Result<T, ColError>;
#[cfg(feature = "python")]
type VariableBuffers = (Py<PyBytes>, Py<PyBytes>);

#[derive(Debug, Error)]
pub enum ColError {
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

#[cfg(feature = "python")]
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

/// Pure-Rust handle to a KohakuVault columnar SQLite database.
pub struct ColumnVault {
    core: Core,
}

impl ColumnVault {
    /// Open or create a column vault at `path` (or `":memory:"`).
    pub fn open(path: &str, read_only: bool) -> ColResult<Self> {
        Ok(Self { core: Core::open(path, read_only)? })
    }

    /// Create a new logical column and return its monotonic `col_id`.
    pub fn create(
        &self,
        name: &str,
        dtype: &str,
        elem_size: usize,
        block_bytes: usize,
    ) -> ColResult<i64> {
        self.core.create(name, dtype, elem_size, block_bytes)
    }

    /// Return the existing `col_id` for `name` or create the column if absent.
    pub fn ensure(
        &self,
        name: &str,
        dtype: &str,
        elem_size: usize,
        block_bytes: usize,
    ) -> ColResult<i64> {
        match self.info(name) {
            Ok((id, existing_dtype, existing_elem_size, _, _)) => {
                if existing_dtype != dtype || existing_elem_size != elem_size {
                    return Err(ColError::Invalid(format!(
                        "column '{name}' already exists with dtype '{existing_dtype}' \
                         (elem_size={existing_elem_size}), requested '{dtype}' \
                         (elem_size={elem_size})"
                    )));
                }
                Ok(id)
            }
            Err(ColError::NotFound) => self.create(name, dtype, elem_size, block_bytes),
            Err(err) => Err(err),
        }
    }

    /// Return `(col_id, dtype, elem_size, length, block_bytes)` for `name`.
    pub fn info(&self, name: &str) -> ColResult<ColumnInfo> {
        self.core.info(name)
    }

    /// Return the committed row count of column `id`.
    pub fn length(&self, id: i64) -> ColResult<usize> {
        self.core.read(|conn| Ok(meta(conn, id)?.length))
    }

    /// List all columns as `(name, dtype, length)` ordered by `col_id`.
    pub fn list_columns(&self) -> ColResult<Vec<(String, String, usize)>> {
        self.core.read(|conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT name, dtype, length FROM kvault_col_columns ORDER BY col_id",
            )?;
            let rows =
                stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? as usize)))?;
            Ok(rows.collect::<Result<Vec<_>, _>>()?)
        })
    }

    /// Delete a column by name. Returns `true` if a column was removed.
    pub fn delete_column(&self, name: &str) -> ColResult<bool> {
        self.core.write(|tx| {
            Ok(tx.execute("DELETE FROM kvault_col_columns WHERE name = ?1", [name])? > 0)
        })
    }

    /// Clear all rows from column `id`.
    pub fn clear(&self, id: i64) -> ColResult<()> {
        self.core.clear(id)
    }

    /// Close primary and pooled connections.
    pub fn close(&self) {
        self.core.close();
    }

    /// Checkpoint WAL passively, returning `(busy, log_frames, checkpointed_frames)`.
    pub fn checkpoint(&self) -> ColResult<(i64, i64, i64)> {
        self.core.checkpoint()
    }

    /// Append raw little-endian fixed-width rows to column `id`.
    pub fn append_fixed(&self, id: i64, data: &[u8]) -> ColResult<()> {
        self.core
            .write(|tx| fixed::append(tx, &meta(tx, id)?, data))
    }

    /// Read a contiguous slice `[start .. start + count)` of fixed-width rows.
    pub fn read_fixed(&self, id: i64, start: usize, count: Option<usize>) -> ColResult<Vec<u8>> {
        self.core
            .read(|conn| fixed::read(conn, &meta(conn, id)?, start, count))
    }

    /// Gather fixed-width rows at arbitrary `indices`.
    pub fn take_fixed(&self, id: i64, indices: &[i64]) -> ColResult<Vec<u8>> {
        self.core
            .read(|conn| fixed::take(conn, &meta(conn, id)?, indices))
    }

    /// Overwrite fixed-width rows starting at `start`.
    pub fn overwrite_fixed(&self, id: i64, start: usize, data: &[u8]) -> ColResult<()> {
        self.core
            .write(|tx| fixed::overwrite(tx, &meta(tx, id)?, start, data))
    }

    /// Replace all fixed-width rows in column `id`.
    pub fn replace_fixed(&self, id: i64, data: &[u8]) -> ColResult<()> {
        self.core.write(|tx| {
            let mut metadata = meta(tx, id)?;
            tx.execute("DELETE FROM kvault_col_blocks WHERE col_id = ?1", [id])?;
            metadata.length = 0;
            fixed::append(tx, &metadata, data)
        })
    }

    /// Append variable-width records given concatenated `data` and little-endian `u64` `offsets`.
    pub fn append_variable(&self, id: i64, data: &[u8], offsets: &[u8]) -> ColResult<()> {
        let positions = varsize_ops::decode_offsets(offsets, data.len())?;
        self.core
            .write(|tx| varsize_ops::append(tx, &meta(tx, id)?, data, &positions))
    }

    /// Append a slice of variable-width byte slices directly.
    pub fn append_variable_slices(&self, id: i64, records: &[&[u8]]) -> ColResult<()> {
        if records.is_empty() {
            return Ok(());
        }
        let total: usize = records.iter().map(|r| r.len()).sum();
        let mut payload = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(records.len() + 1);
        positions.push(0usize);
        for rec in records {
            payload.extend_from_slice(rec);
            positions.push(payload.len());
        }
        self.core
            .write(|tx| varsize_ops::append(tx, &meta(tx, id)?, &payload, &positions))
    }

    /// Read variable-width records `[start .. start + count)` as `(payload, u64_le_offsets)`.
    pub fn read_variable(
        &self,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> ColResult<(Vec<u8>, Vec<u8>)> {
        self.core
            .read(|conn| varsize_ops::read(conn, &meta(conn, id)?, start, count))
    }

    /// Read variable-width records `[start .. start + count)` decoded into individual `Vec<u8>`.
    pub fn read_variable_vec(
        &self,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> ColResult<Vec<Vec<u8>>> {
        let (payload, offsets_bytes) = self.read_variable(id, start, count)?;
        if offsets_bytes.len() < 8 {
            return Ok(Vec::new());
        }
        let n = offsets_bytes.len() / 8;
        let mut out = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n - 1 {
            let a =
                u64::from_le_bytes(offsets_bytes[i * 8..(i + 1) * 8].try_into().unwrap()) as usize;
            let b = u64::from_le_bytes(offsets_bytes[(i + 1) * 8..(i + 2) * 8].try_into().unwrap())
                as usize;
            out.push(payload[a..b].to_vec());
        }
        Ok(out)
    }

    /// Replace all variable-width records in column `id`.
    pub fn replace_variable(&self, id: i64, data: &[u8], offsets: &[u8]) -> ColResult<()> {
        let positions = varsize_ops::decode_offsets(offsets, data.len())?;
        self.core.write(|tx| {
            let mut metadata = meta(tx, id)?;
            tx.execute("DELETE FROM kvault_col_blocks WHERE col_id = ?1", [id])?;
            metadata.length = 0;
            varsize_ops::append(tx, &metadata, data, &positions)
        })
    }

    /// Return `(block_count, used_bytes, capacity_bytes)` for column `id`.
    pub fn storage_info(&self, id: i64) -> ColResult<(usize, usize, usize)> {
        self.core.read(|conn| {
            meta(conn, id)?;
            Ok(conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(used_bytes), 0), COALESCE(SUM(capacity_bytes), 0)
                 FROM kvault_col_blocks WHERE col_id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
                        row.get::<_, i64>(2)? as usize,
                    ))
                },
            )?)
        })
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct _ColumnVault {
    inner: ColumnVault,
}

#[cfg(feature = "python")]
#[pymethods]
impl _ColumnVault {
    #[new]
    #[pyo3(signature = (path, read_only=false))]
    fn new(py: Python<'_>, path: &str, read_only: bool) -> PyResult<Self> {
        let inner = py.allow_threads(|| ColumnVault::open(path, read_only))?;
        Ok(Self { inner })
    }

    fn create(
        &self,
        py: Python<'_>,
        name: &str,
        dtype: &str,
        elem_size: usize,
        block_bytes: usize,
    ) -> PyResult<i64> {
        Ok(py.allow_threads(|| self.inner.create(name, dtype, elem_size, block_bytes))?)
    }

    fn info(&self, py: Python<'_>, name: &str) -> PyResult<ColumnInfo> {
        Ok(py.allow_threads(|| self.inner.info(name))?)
    }

    fn length(&self, py: Python<'_>, id: i64) -> PyResult<usize> {
        Ok(py.allow_threads(|| self.inner.length(id))?)
    }

    fn list_columns(&self, py: Python<'_>) -> PyResult<Vec<(String, String, usize)>> {
        Ok(py.allow_threads(|| self.inner.list_columns())?)
    }

    fn delete_column(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        Ok(py.allow_threads(|| self.inner.delete_column(name))?)
    }

    fn clear(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        Ok(py.allow_threads(|| self.inner.clear(id))?)
    }

    fn close(&self, py: Python<'_>) {
        py.allow_threads(|| self.inner.close());
    }

    fn checkpoint_wal(&self, py: Python<'_>) -> PyResult<(i64, i64, i64)> {
        Ok(py.allow_threads(|| self.inner.checkpoint())?)
    }

    fn append_fixed(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        // Python bytes are immutable and stay alive while the GIL is released.
        let input = data.as_bytes();
        Ok(py.allow_threads(|| self.inner.append_fixed(id, input))?)
    }

    #[pyo3(signature = (id, start=0, count=None))]
    fn read_fixed(
        &self,
        py: Python<'_>,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let output = py.allow_threads(|| self.inner.read_fixed(id, start, count))?;
        Ok(PyBytes::new(py, &output).unbind())
    }

    fn take_fixed(&self, py: Python<'_>, id: i64, indices: Vec<i64>) -> PyResult<Py<PyBytes>> {
        let output = py.allow_threads(|| self.inner.take_fixed(id, &indices))?;
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
        Ok(py.allow_threads(|| self.inner.overwrite_fixed(id, start, input))?)
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
        Ok(py.allow_threads(|| self.inner.append_variable(id, input, encoded))?)
    }

    #[pyo3(signature = (id, start=0, count=None))]
    fn read_variable(
        &self,
        py: Python<'_>,
        id: i64,
        start: usize,
        count: Option<usize>,
    ) -> PyResult<VariableBuffers> {
        let (payload, offsets) = py.allow_threads(|| self.inner.read_variable(id, start, count))?;
        Ok((PyBytes::new(py, &payload).unbind(), PyBytes::new(py, &offsets).unbind()))
    }

    fn replace_fixed(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let input = data.as_bytes();
        Ok(py.allow_threads(|| self.inner.replace_fixed(id, input))?)
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
        Ok(py.allow_threads(|| self.inner.replace_variable(id, input, encoded))?)
    }

    fn storage_info(&self, py: Python<'_>, id: i64) -> PyResult<(usize, usize, usize)> {
        Ok(py.allow_threads(|| self.inner.storage_info(id))?)
    }
}

#[cfg(test)]
mod tests {
    use super::ColumnVault;

    #[test]
    fn fixed_and_variable_roundtrip() {
        let cv = ColumnVault::open(":memory:", false).unwrap();
        let id = cv.ensure("vals", "i64", 8, 64).unwrap();
        let bytes: Vec<u8> = (0i64..20).flat_map(|v| v.to_le_bytes()).collect();
        cv.append_fixed(id, &bytes).unwrap();
        assert_eq!(cv.length(id).unwrap(), 20);

        let slice = cv.read_fixed(id, 15, Some(5)).unwrap();
        let got: Vec<i64> = slice
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![15, 16, 17, 18, 19]);

        let vid = cv.ensure("names", "bytes", 0, 128).unwrap();
        cv.append_variable_slices(vid, &[b"alpha", b"", b"gamma"])
            .unwrap();
        let recs = cv.read_variable_vec(vid, 0, None).unwrap();
        assert_eq!(recs, vec![b"alpha".to_vec(), b"".to_vec(), b"gamma".to_vec()]);
        cv.close();
    }
}
