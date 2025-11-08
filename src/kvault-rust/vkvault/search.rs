//! Search operations for VectorKVault

use crate::vkvault::core::VectorKVault;
use crate::vkvault::metrics::SimilarityMetric;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rusqlite::params;

impl VectorKVault {
    /// Search for k-nearest neighbors
    pub fn search(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        k: usize,
        metric: Option<&str>,
    ) -> PyResult<Vec<(i64, f32, PyObject)>> {
        let query_blob = self.prepare_vector_blob(query_vector)?;

        let metric = if let Some(m) = metric {
            SimilarityMetric::from_str(m).map_err(PyValueError::new_err)?
        } else {
            self.metric
        };

        // Validate metric compatibility
        if !metric.is_compatible_with(self.vector_type) {
            return Err(PyValueError::new_err(format!(
                "Metric '{}' is not compatible with vector type '{}'",
                metric.to_str(),
                self.vector_type.to_str()
            )));
        }

        let conn = self.conn.lock();

        let sql = format!(
            "SELECT v.rowid, v.distance, b.value
             FROM {} v
             JOIN {}_values b ON v.value_ref = b.id
             WHERE v.vector MATCH ? AND k = ?
             ORDER BY v.distance",
            &self.table, &self.table
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {}", e)))?;

        let k_i64 = k as i64;
        let results = stmt
            .query_map(params![query_blob, k_i64], |row| {
                let id: i64 = row.get(0)?;
                let distance: f32 = row.get(1)?;
                let value: Vec<u8> = row.get(2)?;
                Ok((id, distance, value))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        let mut output = Vec::new();
        for result in results {
            let (id, distance, value_bytes) = result
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read row: {}", e)))?;

            // Auto-decode value (EXACTLY SAME AS KVault)
            let decoded_value = self.decode_and_deserialize(py, &value_bytes)?;
            output.push((id, distance, decoded_value));
        }

        Ok(output)
    }

    /// Get value for the most similar vector (KVault-like interface)
    pub fn get(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        metric: Option<&str>,
    ) -> PyResult<PyObject> {
        let results = self.search(py, query_vector, 1, metric)?;

        if results.is_empty() {
            return Err(PyRuntimeError::new_err("No vectors found in database"));
        }

        Ok(results[0].2.clone_ref(py))
    }
}
