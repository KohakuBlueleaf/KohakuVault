//! Vector queries on independent read snapshots.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rusqlite::params;

use super::core::VectorKVault;
use super::metrics::SimilarityMetric;

impl VectorKVault {
    pub fn search(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        k: usize,
        metric: Option<&str>,
    ) -> PyResult<Vec<(i64, f32, PyObject)>> {
        let query_blob = self.prepare_vector_blob(query_vector)?;
        let metric = metric
            .map(SimilarityMetric::from_str)
            .transpose()
            .map_err(PyValueError::new_err)?
            .unwrap_or(self.metric);
        if metric != self.metric {
            return Err(PyValueError::new_err(
                "Search metric must match the stored vec0 index metric",
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(k).map_err(|_| PyValueError::new_err("k is too large"))?;

        let rows: Vec<(i64, f32, Vec<u8>)> = py.allow_threads(|| {
            self.with_reader(|conn| {
                let sql = format!(
                    "SELECT v.rowid, v.distance, b.value FROM {} v
                 JOIN {}_values b ON v.value_ref = b.id
                 WHERE v.vector MATCH ?1 AND k = ?2 ORDER BY v.distance",
                    self.table, self.table,
                );
                let mut statement = conn.prepare_cached(&sql)?;
                let rows = statement.query_map(params![query_blob, limit], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?;
                rows.collect()
            })
        })?;

        // Decode only after releasing the SQLite snapshot and connection.
        rows.into_iter()
            .map(|(id, distance, value)| {
                Ok((id, distance, self.decode_and_deserialize(py, &value)?))
            })
            .collect()
    }

    pub fn get(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        metric: Option<&str>,
    ) -> PyResult<PyObject> {
        self.search(py, query_vector, 1, metric)?
            .into_iter()
            .next()
            .map(|(_, _, value)| value)
            .ok_or_else(|| PyRuntimeError::new_err("No vectors found in database"))
    }
}
