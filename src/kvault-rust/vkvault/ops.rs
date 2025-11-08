//! CRUD operations for VectorKVault

use crate::vkvault::core::VectorKVault;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, OptionalExtension};

impl VectorKVault {
    /// Insert a vector-value pair
    pub fn insert(
        &self,
        _py: Python<'_>,
        vector: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        _metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<i64> {
        // Convert vector to blob
        let vector_blob = self.prepare_vector_blob(vector)?;

        // Convert value to bytes
        let value_bytes = self.extract_bytes(value)?;

        let conn = self.conn.lock();

        // Insert value into blob table
        let value_id: i64 = conn
            .query_row(
                &format!("INSERT INTO {}_values (value) VALUES (?) RETURNING id", &self.table),
                params![value_bytes],
                |row| row.get(0),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert value: {}", e)))?;

        // Insert vector into vec0 table
        conn.execute(
            &format!("INSERT INTO {} (vector, value_ref) VALUES (?, ?)", &self.table),
            params![vector_blob, value_id],
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert vector: {}", e)))?;

        // Get the rowid of the inserted vector
        let vector_id: i64 = conn.last_insert_rowid();

        Ok(vector_id)
    }

    /// Get vector and value by ID
    pub fn get_by_id(&self, py: Python<'_>, id: i64) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let conn = self.conn.lock();

        let sql = format!(
            "SELECT v.vector, b.value
             FROM {} v
             JOIN {}_values b ON v.value_ref = b.id
             WHERE v.rowid = ?",
            &self.table, &self.table
        );

        let result: Option<(Vec<u8>, Vec<u8>)> = conn
            .query_row(&sql, params![id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        match result {
            Some((vector, value)) => {
                let vector_py = PyBytes::new_bound(py, &vector).unbind();
                let value_py = PyBytes::new_bound(py, &value).unbind();
                Ok((vector_py, value_py))
            }
            None => Err(PyRuntimeError::new_err(format!("ID {} not found", id))),
        }
    }

    /// Delete by ID
    pub fn delete(&self, id: i64) -> PyResult<()> {
        let conn = self.conn.lock();

        // Get value_ref before deleting
        let value_ref: Option<i64> = conn
            .query_row(
                &format!("SELECT value_ref FROM {} WHERE rowid = ?", &self.table),
                params![id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get value_ref: {}", e)))?;

        if let Some(value_id) = value_ref {
            // Delete from vec0 table
            conn.execute(&format!("DELETE FROM {} WHERE rowid = ?", &self.table), params![id])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {}", e)))?;

            // Delete from values table
            conn.execute(
                &format!("DELETE FROM {}_values WHERE id = ?", &self.table),
                params![value_id],
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete value: {}", e)))?;
        }

        Ok(())
    }

    /// Update vector or value by ID
    pub fn update(
        &self,
        id: i64,
        vector: Option<&Bound<'_, PyAny>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if vector.is_none() && value.is_none() {
            return Err(PyValueError::new_err("Must provide either vector or value to update"));
        }

        let conn = self.conn.lock();

        // Update vector if provided
        if let Some(vec) = vector {
            let vector_blob = self.prepare_vector_blob(vec)?;
            conn.execute(
                &format!("UPDATE {} SET vector = ? WHERE rowid = ?", &self.table),
                params![vector_blob, id],
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update vector: {}", e)))?;
        }

        // Update value if provided
        if let Some(val) = value {
            let value_bytes = self.extract_bytes(val)?;

            // Get value_ref
            let value_ref: i64 = conn
                .query_row(
                    &format!("SELECT value_ref FROM {} WHERE rowid = ?", &self.table),
                    params![id],
                    |row| row.get(0),
                )
                .map_err(|e| PyRuntimeError::new_err(format!("ID {} not found: {}", id, e)))?;

            conn.execute(
                &format!("UPDATE {}_values SET value = ? WHERE id = ?", &self.table),
                params![value_bytes, value_ref],
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update value: {}", e)))?;
        }

        Ok(())
    }
}
