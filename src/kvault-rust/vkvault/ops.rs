//! Atomic vector/payload mutations with Python conversion outside database locks.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use super::core::VectorKVault;

fn sql_error(error: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

impl VectorKVault {
    fn pack_value(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let packer = *self.auto_packer.lock();
        if let Some(packer) = packer {
            let (bytes, encoding) = packer.serialize(py, value)?;
            Ok(self.encode_value(&bytes, encoding))
        } else {
            self.extract_bytes(value)
        }
    }

    pub fn insert(
        &self,
        py: Python<'_>,
        vector: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        _metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<i64> {
        self.ensure_writable()?;
        let vector_blob = self.prepare_vector_blob(vector)?;
        let value_bytes = self.pack_value(py, value)?;

        py.allow_threads(|| {
            let mut conn = self.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(sql_error)?;
            let value_id: i64 = tx
                .query_row(
                    &format!("INSERT INTO {}_values (value) VALUES (?1) RETURNING id", self.table),
                    params![value_bytes],
                    |row| row.get(0),
                )
                .map_err(sql_error)?;

            // Both tables become visible together, or neither is published.
            tx.execute(
                &format!("INSERT INTO {} (vector, value_ref) VALUES (?1, ?2)", self.table),
                params![vector_blob, value_id],
            )
            .map_err(sql_error)?;
            let id = tx.last_insert_rowid();
            tx.commit().map_err(sql_error)?;
            Ok(id)
        })
    }

    pub fn get_by_id(&self, py: Python<'_>, id: i64) -> PyResult<(Py<PyBytes>, PyObject)> {
        let result: Option<(Vec<u8>, Vec<u8>)> = py.allow_threads(|| {
            self.with_reader(|conn| {
                let sql = format!(
                    "SELECT v.vector, b.value FROM {} v
                 JOIN {}_values b ON v.value_ref = b.id WHERE v.rowid = ?1",
                    self.table, self.table,
                );
                conn.prepare_cached(&sql)?
                    .query_row([id], |r| Ok((r.get(0)?, r.get(1)?)))
                    .optional()
            })
        })?;
        let (vector, value) =
            result.ok_or_else(|| PyRuntimeError::new_err(format!("ID {id} not found")))?;
        Ok((PyBytes::new(py, &vector).unbind(), self.decode_and_deserialize(py, &value)?))
    }

    pub fn delete(&self, id: i64) -> PyResult<()> {
        self.ensure_writable()?;
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql_error)?;
        let value_id: Option<i64> = tx
            .query_row(
                &format!("SELECT value_ref FROM {} WHERE rowid = ?1", self.table),
                [id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql_error)?;

        if let Some(value_id) = value_id {
            tx.execute(&format!("DELETE FROM {} WHERE rowid = ?1", self.table), [id])
                .map_err(sql_error)?;
            tx.execute(&format!("DELETE FROM {}_values WHERE id = ?1", self.table), [value_id])
                .map_err(sql_error)?;
        }
        tx.commit().map_err(sql_error)
    }

    pub fn update(
        &self,
        py: Python<'_>,
        id: i64,
        vector: Option<&Bound<'_, PyAny>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        if vector.is_none() && value.is_none() {
            return Err(PyValueError::new_err("Must provide either vector or value to update"));
        }
        let vector_blob = vector
            .map(|value| self.prepare_vector_blob(value))
            .transpose()?;
        let value_bytes = value.map(|value| self.pack_value(py, value)).transpose()?;

        py.allow_threads(|| {
            let mut conn = self.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(sql_error)?;
            let value_id: i64 = tx
                .query_row(
                    &format!("SELECT value_ref FROM {} WHERE rowid = ?1", self.table),
                    [id],
                    |r| r.get(0),
                )
                .map_err(|error| PyRuntimeError::new_err(format!("ID {id} not found: {error}")))?;

            if let Some(vector_blob) = vector_blob {
                tx.execute(
                    &format!("UPDATE {} SET vector = ?1 WHERE rowid = ?2", self.table),
                    params![vector_blob, id],
                )
                .map_err(sql_error)?;
            }
            if let Some(value_bytes) = value_bytes {
                tx.execute(
                    &format!("UPDATE {}_values SET value = ?1 WHERE id = ?2", self.table),
                    params![value_bytes, value_id],
                )
                .map_err(sql_error)?;
            }
            tx.commit().map_err(sql_error)
        })
    }
}
