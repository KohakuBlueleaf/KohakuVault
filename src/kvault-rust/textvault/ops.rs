//! CRUD operations for TextVault

use crate::textvault::core::TextVault;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rusqlite::{params, OptionalExtension};

impl TextVault {
    /// Insert a document with text content and value
    ///
    /// For single-column TextVault: texts can be a string
    /// For multi-column TextVault: texts should be a dict mapping column names to values
    pub fn insert(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        _metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<i64> {
        // Parse text inputs based on column configuration
        let text_values = self.parse_text_input(texts)?;

        // Convert value to bytes using auto-packing (EXACTLY SAME AS KVault/VectorKVault)
        let value_bytes = {
            let auto_pack_guard = self.auto_packer.lock();
            if let Some(ref packer) = *auto_pack_guard {
                // Auto-pack mode: serialize automatically
                let (serialized_bytes, encoding) = packer.serialize(py, value)?;
                drop(auto_pack_guard);

                // Add header if encoding is not Raw
                self.encode_value(&serialized_bytes, encoding)
            } else {
                // Legacy mode: must be bytes
                drop(auto_pack_guard);
                self.extract_bytes(value)?
            }
        };

        let conn = self.conn.lock();

        // Insert value into blob table
        let value_id: i64 = conn
            .query_row(
                &format!("INSERT INTO {}_values (value) VALUES (?) RETURNING id", &self.table),
                params![value_bytes],
                |row| row.get(0),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert value: {}", e)))?;

        // Build INSERT statement for FTS5 table
        let col_names: Vec<&str> = self.columns.iter().map(|s| s.as_str()).collect();
        let placeholders: Vec<&str> = col_names.iter().map(|_| "?").collect();

        let sql = format!(
            "INSERT INTO {} ({}, value_ref) VALUES ({}, ?)",
            &self.table,
            col_names.join(", "),
            placeholders.join(", ")
        );

        // Build parameters: text values + value_ref
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to prepare insert statement: {}", e))
        })?;

        // Execute with text values and value_ref
        let mut param_values: Vec<rusqlite::types::Value> = text_values
            .iter()
            .map(|s| rusqlite::types::Value::Text(s.clone()))
            .collect();
        param_values.push(rusqlite::types::Value::Integer(value_id));

        stmt.execute(rusqlite::params_from_iter(param_values))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to insert into FTS5 table: {}", e))
            })?;

        // Get the rowid of the inserted document
        let doc_id: i64 = conn.last_insert_rowid();

        Ok(doc_id)
    }

    /// Parse text input based on column configuration
    ///
    /// Single column: accepts string directly
    /// Multiple columns: accepts dict mapping column names to values
    fn parse_text_input(&self, texts: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        if self.columns.len() == 1 {
            // Single column: accept string directly or dict with one key
            if let Ok(text) = texts.extract::<String>() {
                return Ok(vec![text]);
            }
            if let Ok(dict) = texts.downcast::<PyDict>() {
                if let Some(value) = dict.get_item(&self.columns[0])? {
                    return Ok(vec![value.extract::<String>()?]);
                }
            }
            return Err(PyValueError::new_err(format!(
                "Expected string or dict with key '{}' for single-column TextVault",
                &self.columns[0]
            )));
        }

        // Multiple columns: must be a dict
        let dict = texts.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "Expected dict with keys {:?} for multi-column TextVault",
                &self.columns
            ))
        })?;

        let mut values = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let value = dict
                .get_item(col)?
                .ok_or_else(|| PyValueError::new_err(format!("Missing column '{}' in input", col)))?
                .extract::<String>()?;
            values.push(value);
        }

        Ok(values)
    }

    /// Get document text and value by ID
    pub fn get_by_id(&self, py: Python<'_>, id: i64) -> PyResult<(PyObject, PyObject)> {
        let conn = self.conn.lock();

        // Build SELECT for all text columns
        let col_names = self.columns.join(", ");
        let sql = format!(
            "SELECT {}, b.value
             FROM {} t
             JOIN {}_values b ON t.value_ref = b.id
             WHERE t.rowid = ?",
            col_names, &self.table, &self.table
        );

        let result: Option<(Vec<String>, Vec<u8>)> = conn
            .query_row(&sql, params![id], |row| {
                let mut texts = Vec::with_capacity(self.columns.len());
                for i in 0..self.columns.len() {
                    texts.push(row.get::<_, String>(i)?);
                }
                let value: Vec<u8> = row.get(self.columns.len())?;
                Ok((texts, value))
            })
            .optional()
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        match result {
            Some((texts, value_bytes)) => {
                // Return texts as dict if multiple columns, string if single column
                let texts_py = if self.columns.len() == 1 {
                    texts[0].clone().into_py(py)
                } else {
                    let dict = PyDict::new_bound(py);
                    for (col, text) in self.columns.iter().zip(texts.iter()) {
                        dict.set_item(col, text)?;
                    }
                    dict.into_py(py)
                };

                // Auto-decode value (EXACTLY SAME AS KVault/VectorKVault)
                let value_py = self.decode_and_deserialize(py, &value_bytes)?;

                Ok((texts_py, value_py))
            }
            None => Err(PyRuntimeError::new_err(format!("ID {} not found", id))),
        }
    }

    /// Get value by exact key match (single column only)
    ///
    /// Uses FTS5 exact phrase matching with quotes
    pub fn get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        if self.columns.len() != 1 {
            return Err(PyValueError::new_err(
                "get() with string key only works for single-column TextVault. Use get_by_id() or search() instead.",
            ));
        }

        let conn = self.conn.lock();

        // Use exact phrase matching with escaped quotes
        let escaped_key = key.replace('"', "\"\"");
        let sql = format!(
            "SELECT b.value
             FROM {} t
             JOIN {}_values b ON t.value_ref = b.id
             WHERE t.{} MATCH '\"{}\"'
             LIMIT 1",
            &self.table, &self.table, &self.columns[0], escaped_key
        );

        let result: Option<Vec<u8>> = conn
            .query_row(&sql, [], |row| row.get(0))
            .optional()
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        match result {
            Some(value_bytes) => self.decode_and_deserialize(py, &value_bytes),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!("Key not found: {}", key))),
        }
    }

    /// Delete document by ID
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
            // Delete from FTS5 table
            conn.execute(&format!("DELETE FROM {} WHERE rowid = ?", &self.table), params![id])
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to delete document: {}", e))
                })?;

            // Delete from values table
            conn.execute(
                &format!("DELETE FROM {}_values WHERE id = ?", &self.table),
                params![value_id],
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete value: {}", e)))?;
        }

        Ok(())
    }

    /// Update document text or value by ID
    pub fn update(
        &self,
        py: Python<'_>,
        id: i64,
        texts: Option<&Bound<'_, PyAny>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if texts.is_none() && value.is_none() {
            return Err(PyValueError::new_err("Must provide either texts or value to update"));
        }

        let conn = self.conn.lock();

        // Update texts if provided
        if let Some(text_input) = texts {
            let text_values = self.parse_text_input(text_input)?;

            // Build UPDATE statement
            let set_clauses: Vec<String> = self
                .columns
                .iter()
                .map(|col| format!("{} = ?", col))
                .collect();

            let sql =
                format!("UPDATE {} SET {} WHERE rowid = ?", &self.table, set_clauses.join(", "));

            let mut param_values: Vec<rusqlite::types::Value> = text_values
                .iter()
                .map(|s| rusqlite::types::Value::Text(s.clone()))
                .collect();
            param_values.push(rusqlite::types::Value::Integer(id));

            conn.execute(&sql, rusqlite::params_from_iter(param_values))
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to update document: {}", e))
                })?;
        }

        // Update value if provided
        if let Some(val) = value {
            // Convert value to bytes using auto-packing
            let value_bytes = {
                let auto_pack_guard = self.auto_packer.lock();
                if let Some(ref packer) = *auto_pack_guard {
                    let (serialized_bytes, encoding) = packer.serialize(py, val)?;
                    drop(auto_pack_guard);
                    self.encode_value(&serialized_bytes, encoding)
                } else {
                    drop(auto_pack_guard);
                    self.extract_bytes(val)?
                }
            };

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

    /// Clear all documents from the vault
    pub fn clear(&self) -> PyResult<()> {
        let conn = self.conn.lock();

        // Delete from FTS5 table
        conn.execute(&format!("DELETE FROM {}", &self.table), [])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to clear FTS5 table: {}", e)))?;

        // Delete from values table
        conn.execute(&format!("DELETE FROM {}_values", &self.table), [])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to clear values table: {}", e)))?;

        Ok(())
    }

    /// Get rowids (for iteration) with optional limit and offset
    pub fn keys(&self, limit: Option<i64>, offset: Option<i64>) -> PyResult<Vec<i64>> {
        let conn = self.conn.lock();

        let sql = match (limit, offset) {
            (Some(lim), Some(off)) => {
                format!("SELECT rowid FROM {} LIMIT {} OFFSET {}", &self.table, lim, off)
            }
            (Some(lim), None) => {
                format!("SELECT rowid FROM {} LIMIT {}", &self.table, lim)
            }
            (None, Some(off)) => {
                format!("SELECT rowid FROM {} LIMIT -1 OFFSET {}", &self.table, off)
            }
            (None, None) => {
                format!("SELECT rowid FROM {}", &self.table)
            }
        };

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {}", e)))?;

        let ids: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }
}
