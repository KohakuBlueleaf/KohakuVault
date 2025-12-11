//! TextVault - Full-text search storage with BM25 ranking
//!
//! This module provides a KVault-like interface with full-text search capabilities.
//! Built on top of SQLite's FTS5 extension.
//!
//! Features:
//! - Full-text search with BM25 ranking
//! - Exact match key-value lookups
//! - Flexible multi-column schema
//! - Auto-packing for arbitrary Python values

mod core;
mod ops;
mod search;

use core::TextVault;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// TextVault - Full-text search with arbitrary values
///
/// Schema:
/// - FTS5 virtual table: stores indexed text and value references
/// - separate blob table: stores actual values
///
/// Example:
/// ```python
/// tv = TextVault("data.db", columns=["title", "body"])
/// doc_id = tv.insert({"title": "Hello", "body": "World"}, document_bytes)
/// results = tv.search("hello", k=10)
/// ```
#[pyclass(name = "_TextVault")]
pub struct _TextVault {
    inner: TextVault,
}

#[pymethods]
impl _TextVault {
    /// Create or open a TextVault
    ///
    /// Args:
    ///     path: Database file path
    ///     table: Table name (default: "text_vault")
    ///     columns: List of indexed text columns (default: ["content"])
    #[new]
    #[pyo3(signature = (path, table="text_vault", columns=None))]
    fn new(path: &str, table: &str, columns: Option<Vec<String>>) -> PyResult<Self> {
        Ok(Self { inner: TextVault::new(path, table, columns)? })
    }

    /// Insert a document with text content and value
    ///
    /// Args:
    ///     texts: Text content - string for single column, dict for multiple columns
    ///     value: Value to store (bytes-like, or any object when auto-pack enabled)
    ///     metadata: Optional dict of metadata (not yet supported)
    ///
    /// Returns:
    ///     int: ID of inserted document
    #[pyo3(signature = (texts, value, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<i64> {
        self.inner.insert(py, texts, value, metadata)
    }

    /// Search documents using FTS5 with BM25 ranking
    ///
    /// Args:
    ///     query: FTS5 query string (supports standard FTS5 query syntax)
    ///     k: Maximum number of results to return (default: 10)
    ///     column: Optional column to search in (for multi-column vaults)
    ///     escape: If true, escape special FTS5 characters for literal matching (default: true)
    ///             Set to false to use raw FTS5 query syntax (AND, OR, NOT, *, etc.)
    ///
    /// Returns:
    ///     List of (id, bm25_score, value) tuples, sorted by relevance
    ///     Higher scores indicate more relevant results
    #[pyo3(signature = (query, k=10, column=None, escape=true))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
        column: Option<&str>,
        escape: bool,
    ) -> PyResult<Vec<(i64, f64, PyObject)>> {
        self.inner.search(py, query, k, column, escape)
    }

    /// Search and return documents with their text content
    ///
    /// Returns:
    ///     List of (id, bm25_score, texts, value) tuples
    #[pyo3(signature = (query, k=10, column=None, escape=true))]
    fn search_with_text(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
        column: Option<&str>,
        escape: bool,
    ) -> PyResult<Vec<(i64, f64, PyObject, PyObject)>> {
        self.inner.search_with_text(py, query, k, column, escape)
    }

    /// Search with highlighted snippets
    ///
    /// Args:
    ///     query: FTS5 query string
    ///     k: Maximum number of results
    ///     snippet_column: Column to generate snippet from (default: first column)
    ///     snippet_tokens: Number of tokens around match (default: 10)
    ///     highlight_start: Highlight start marker (default: "**")
    ///     highlight_end: Highlight end marker (default: "**")
    ///     escape: If true, escape special FTS5 characters (default: true)
    ///
    /// Returns:
    ///     List of (id, bm25_score, snippet, value) tuples
    #[pyo3(signature = (query, k=10, snippet_column=None, snippet_tokens=None, highlight_start=None, highlight_end=None, escape=true))]
    #[allow(clippy::too_many_arguments)]
    fn search_with_snippets(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
        snippet_column: Option<&str>,
        snippet_tokens: Option<i32>,
        highlight_start: Option<&str>,
        highlight_end: Option<&str>,
        escape: bool,
    ) -> PyResult<Vec<(i64, f64, String, PyObject)>> {
        self.inner.search_with_snippets(
            py,
            query,
            k,
            snippet_column,
            snippet_tokens,
            highlight_start,
            highlight_end,
            escape,
        )
    }

    /// Get value by exact key match (single-column TextVault only)
    ///
    /// Args:
    ///     key: Exact text to match
    ///
    /// Returns:
    ///     Value associated with the key (auto-decoded)
    ///
    /// Raises:
    ///     KeyError: If key not found
    ///     ValueError: If vault has multiple columns
    #[pyo3(signature = (key))]
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.inner.get(py, key)
    }

    /// Get document text and value by ID
    ///
    /// Args:
    ///     id: Row ID
    ///
    /// Returns:
    ///     (texts, value) tuple where texts is string (single column) or dict (multiple columns)
    ///     Value is auto-decoded
    ///
    /// Raises:
    ///     RuntimeError: If ID doesn't exist
    #[pyo3(signature = (id))]
    fn get_by_id(&self, py: Python<'_>, id: i64) -> PyResult<(PyObject, PyObject)> {
        self.inner.get_by_id(py, id)
    }

    /// Delete document by ID
    ///
    /// Args:
    ///     id: Row ID to delete
    #[pyo3(signature = (id))]
    fn delete(&self, id: i64) -> PyResult<()> {
        self.inner.delete(id)
    }

    /// Update document text or value by ID
    ///
    /// Args:
    ///     id: Row ID
    ///     texts: New text content (optional)
    ///     value: New value (optional)
    #[pyo3(signature = (id, texts=None, value=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: i64,
        texts: Option<&Bound<'_, PyAny>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.inner.update(py, id, texts, value)
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

    /// Get total count of documents
    ///
    /// Returns:
    ///     int: Number of documents
    fn count(&self) -> PyResult<i64> {
        self.inner.count()
    }

    /// Count documents matching a query
    ///
    /// Args:
    ///     query: FTS5 query string
    ///
    /// Returns:
    ///     int: Number of matching documents
    #[pyo3(signature = (query))]
    fn count_matches(&self, query: &str) -> PyResult<i64> {
        self.inner.count_matches(query)
    }

    /// Get TextVault info
    ///
    /// Returns:
    ///     dict: Table info (table, columns, count)
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.inner.info(py)
    }

    /// Get column names
    ///
    /// Returns:
    ///     list: Column names
    fn columns(&self) -> Vec<String> {
        self.inner.get_columns().to_vec()
    }

    /// Clear all documents
    fn clear(&self) -> PyResult<()> {
        self.inner.clear()
    }

    /// Get document IDs with optional pagination
    ///
    /// Args:
    ///     limit: Maximum number of IDs to return (default: no limit)
    ///     offset: Number of IDs to skip (default: 0)
    ///
    /// Returns:
    ///     list: Rowids in the vault
    #[pyo3(signature = (limit=None, offset=None))]
    fn keys(&self, limit: Option<i64>, offset: Option<i64>) -> PyResult<Vec<i64>> {
        self.inner.keys(limit, offset)
    }

    // ----------------------------
    // Auto-packing Management (EXACTLY SAME AS KVault/VectorKVault)
    // ----------------------------

    /// Enable auto-packing (allows arbitrary Python objects as values)
    ///
    /// When enabled, TextVault automatically serializes Python objects:
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

/// Register TextVault types with Python module
pub fn register_textvault_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_TextVault>()?;
    Ok(())
}
