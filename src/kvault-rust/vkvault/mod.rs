//! VectorKVault - Vector similarity search storage
//!
//! This module provides a KVault-like interface with vector keys and similarity search.
//! Built on top of sqlite-vec's vec0 virtual table.

mod core;
mod metrics;
mod ops;
mod search;

use core::VectorKVault;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// VectorKVault - Vector similarity search with arbitrary values
///
/// Schema:
/// - vec0 virtual table: stores vectors and value references
/// - separate blob table: stores actual values
///
/// Example:
/// ```python
/// vkv = VectorKVault("data.db", dimensions=768, metric="cosine")
/// doc_id = vkv.insert(embedding, document_bytes)
/// results = vkv.search(query_embedding, k=10)
/// ```
#[pyclass(name = "_VectorKVault")]
pub struct _VectorKVault {
    inner: VectorKVault,
}

#[pymethods]
impl _VectorKVault {
    /// Create or open a VectorKVault
    ///
    /// Args:
    ///     path: Database file path
    ///     table: Table name (default: "vec_kvault")
    ///     dimensions: Vector dimensions (required)
    ///     metric: Distance metric - "cosine", "l2", "l1", or "hamming" (default: "cosine")
    ///     vector_type: Element type - "f32", "int8", or "bit" (default: "f32")
    #[new]
    #[pyo3(signature = (path, table="vec_kvault", dimensions=768, metric="cosine", vector_type="f32"))]
    fn new(
        path: &str,
        table: &str,
        dimensions: usize,
        metric: &str,
        vector_type: &str,
    ) -> PyResult<Self> {
        Ok(Self { inner: VectorKVault::new(path, table, dimensions, metric, vector_type)? })
    }

    /// Insert a vector-value pair
    ///
    /// Args:
    ///     vector: Vector (list of floats, numpy array, or bytes)
    ///     value: Value to store (bytes-like)
    ///     metadata: Optional dict of metadata (not yet supported)
    ///
    /// Returns:
    ///     int: ID of inserted item
    #[pyo3(signature = (vector, value, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        vector: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<i64> {
        self.inner.insert(py, vector, value, metadata)
    }

    /// Search for k-nearest neighbors
    ///
    /// Args:
    ///     query_vector: Query vector
    ///     k: Number of results to return (default: 10)
    ///     metric: Override distance metric (optional)
    ///
    /// Returns:
    ///     List of (id, distance, value) tuples (values are auto-decoded)
    #[pyo3(signature = (query_vector, k=10, metric=None))]
    fn search(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        k: usize,
        metric: Option<&str>,
    ) -> PyResult<Vec<(i64, f32, PyObject)>> {
        self.inner.search(py, query_vector, k, metric)
    }

    /// Get value for the most similar vector (KVault-like interface)
    ///
    /// Args:
    ///     query_vector: Query vector
    ///     metric: Override distance metric (optional)
    ///
    /// Returns:
    ///     Value of the most similar vector (auto-decoded)
    ///
    /// Raises:
    ///     RuntimeError: If no vectors found
    #[pyo3(signature = (query_vector, metric=None))]
    fn get(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        metric: Option<&str>,
    ) -> PyResult<PyObject> {
        self.inner.get(py, query_vector, metric)
    }

    /// Get vector and value by ID
    ///
    /// Args:
    ///     id: Row ID
    ///
    /// Returns:
    ///     (vector, value) tuple (value is auto-decoded)
    #[pyo3(signature = (id))]
    fn get_by_id(&self, py: Python<'_>, id: i64) -> PyResult<(Py<PyBytes>, PyObject)> {
        self.inner.get_by_id(py, id)
    }

    /// Delete by ID
    ///
    /// Args:
    ///     id: Row ID to delete
    #[pyo3(signature = (id))]
    fn delete(&self, id: i64) -> PyResult<()> {
        self.inner.delete(id)
    }

    /// Update vector or value by ID
    ///
    /// Args:
    ///     id: Row ID
    ///     vector: New vector (optional)
    ///     value: New value (optional)
    #[pyo3(signature = (id, vector=None, value=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: i64,
        vector: Option<&Bound<'_, PyAny>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.inner.update(py, id, vector, value)
    }

    /// Check if ID exists
    ///
    /// Args:
    ///     id: Row ID
    ///
    /// Returns:
    ///     bool: True if exists
    #[pyo3(signature = (id))]
    fn exists(&self, id: i64) -> PyResult<bool> {
        self.inner.exists(id)
    }

    /// Get total count of vectors
    ///
    /// Returns:
    ///     int: Number of vectors
    fn count(&self) -> PyResult<i64> {
        self.inner.count()
    }

    /// Get VectorKVault info
    ///
    /// Returns:
    ///     dict: Table info (dimensions, metric, vector_type, count)
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.inner.info(py)
    }

    /// Enable auto-packing (allows arbitrary Python objects as values)
    ///
    /// When enabled, VectorKVault automatically serializes Python objects:
    /// - numpy arrays → DataPacker vec:*
    /// - dicts/lists → MessagePack
    /// - int/float → DataPacker i64/f64
    /// - bytes → Raw (no header)
    /// - Custom objects → Pickle (if use_pickle=True)
    ///
    /// Args:
    ///     use_pickle: Allow pickle for custom objects as last resort (default: True)
    #[pyo3(signature = (use_pickle=true))]
    fn enable_auto_pack(&self, use_pickle: bool) -> PyResult<()> {
        self.inner.enable_auto_pack(use_pickle)
    }

    /// Disable auto-packing (return to bytes-only mode)
    fn disable_auto_pack(&self) {
        self.inner.disable_auto_pack()
    }

    /// Check if auto-packing is enabled
    ///
    /// Returns:
    ///     bool: True if auto-packing is enabled
    fn auto_pack_enabled(&self) -> bool {
        self.inner.auto_pack_enabled()
    }

    /// Enable header format for new writes
    fn enable_headers(&self) {
        self.inner.enable_headers()
    }

    /// Disable header format (return to raw bytes mode)
    fn disable_headers(&self) {
        self.inner.disable_headers()
    }

    /// Check if headers are enabled
    ///
    /// Returns:
    ///     bool: True if headers are enabled
    fn headers_enabled(&self) -> bool {
        self.inner.headers_enabled()
    }
}

/// Register VectorKVault types with Python module
pub fn register_vkvault_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_VectorKVault>()?;
    Ok(())
}
