//! Search operations for TextVault
//!
//! Provides full-text search using FTS5 with BM25 ranking.

use crate::textvault::core::TextVault;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rusqlite::params;

/// Escape a query string for safe FTS5 literal/phrase matching.
///
/// FTS5 uses double quotes for phrase queries. To include literal special
/// characters (?, +, @, etc.), the query must be wrapped in double quotes.
/// Internal double quotes are escaped by doubling them ("").
///
/// Example:
///   "hello world" -> "\"hello world\""
///   "What is this?" -> "\"What is this?\""
///   "He said \"hi\"" -> "\"He said \"\"hi\"\"\""
fn escape_fts5_query(query: &str) -> String {
    // If query is empty, return empty
    if query.trim().is_empty() {
        return String::new();
    }

    // Escape internal double quotes by doubling them, then wrap in quotes
    let escaped = query.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

impl TextVault {
    /// Search documents using FTS5 with BM25 ranking
    ///
    /// Args:
    ///     query: FTS5 query string (supports standard FTS5 query syntax)
    ///     k: Maximum number of results to return (default: 10)
    ///     column: Optional column to search in (for multi-column vaults)
    ///     escape: If true, escape special FTS5 characters for safe literal matching (default: true)
    ///
    /// Returns:
    ///     List of (id, bm25_score, value) tuples, sorted by relevance (best first)
    ///
    /// Note: BM25 scores are negative in SQLite FTS5 (more negative = more relevant)
    /// We return them as positive scores (higher = more relevant) for consistency
    pub fn search(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
        column: Option<&str>,
        escape: bool,
    ) -> PyResult<Vec<(i64, f64, PyObject)>> {
        let conn = self.conn.lock();

        // Escape query if requested (default: true for safety)
        let safe_query = if escape {
            escape_fts5_query(query)
        } else {
            query.to_string()
        };

        // Handle empty query after escaping
        if safe_query.is_empty() {
            return Ok(Vec::new());
        }

        // Build FTS5 query with optional column prefix
        let fts_query = if let Some(col) = column {
            // Validate column exists
            if !self.columns.contains(&col.to_string()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Column '{}' not found. Available columns: {:?}",
                    col, &self.columns
                )));
            }
            format!("{}:{}", col, safe_query)
        } else {
            safe_query
        };

        // Use bm25() function for ranking
        // FTS5 bm25() returns negative scores (more negative = more relevant)
        // Note: FTS5 tables require using table name directly for rowid, not aliases
        let sql = format!(
            "SELECT {table}.rowid, bm25({table}), {table}_values.value
             FROM {table}
             JOIN {table}_values ON {table}.value_ref = {table}_values.id
             WHERE {table} MATCH ?
             ORDER BY bm25({table})
             LIMIT ?",
            table = &self.table
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {}", e)))?;

        let k_i64 = k as i64;
        let results = stmt
            .query_map(params![fts_query, k_i64], |row| {
                let id: i64 = row.get(0)?;
                let score: f64 = row.get(1)?;
                let value: Vec<u8> = row.get(2)?;
                Ok((id, score, value))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        let mut output = Vec::new();
        for result in results {
            let (id, score, value_bytes) = result
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read row: {}", e)))?;

            // Convert BM25 score to positive (more relevant = higher score)
            let positive_score = -score;

            // Auto-decode value (EXACTLY SAME AS KVault/VectorKVault)
            let decoded_value = self.decode_and_deserialize(py, &value_bytes)?;
            output.push((id, positive_score, decoded_value));
        }

        Ok(output)
    }

    /// Search and return documents with their text content
    ///
    /// Returns:
    ///     List of (id, bm25_score, texts, value) tuples
    pub fn search_with_text(
        &self,
        py: Python<'_>,
        query: &str,
        k: usize,
        column: Option<&str>,
        escape: bool,
    ) -> PyResult<Vec<(i64, f64, PyObject, PyObject)>> {
        let conn = self.conn.lock();

        // Escape query if requested
        let safe_query = if escape {
            escape_fts5_query(query)
        } else {
            query.to_string()
        };

        if safe_query.is_empty() {
            return Ok(Vec::new());
        }

        // Build FTS5 query with optional column prefix
        let fts_query = if let Some(col) = column {
            if !self.columns.contains(&col.to_string()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Column '{}' not found. Available columns: {:?}",
                    col, &self.columns
                )));
            }
            format!("{}:{}", col, safe_query)
        } else {
            safe_query
        };

        // Build SELECT for all text columns with table prefix
        let col_names: Vec<String> = self
            .columns
            .iter()
            .map(|c| format!("{}.{}", &self.table, c))
            .collect();
        let col_select = col_names.join(", ");
        let sql = format!(
            "SELECT {table}.rowid, bm25({table}), {cols}, {table}_values.value
             FROM {table}
             JOIN {table}_values ON {table}.value_ref = {table}_values.id
             WHERE {table} MATCH ?
             ORDER BY bm25({table})
             LIMIT ?",
            table = &self.table,
            cols = col_select
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {}", e)))?;

        let num_cols = self.columns.len();
        let k_i64 = k as i64;
        let results = stmt
            .query_map(params![fts_query, k_i64], |row| {
                let id: i64 = row.get(0)?;
                let score: f64 = row.get(1)?;
                let mut texts = Vec::with_capacity(num_cols);
                for i in 0..num_cols {
                    texts.push(row.get::<_, String>(2 + i)?);
                }
                let value: Vec<u8> = row.get(2 + num_cols)?;
                Ok((id, score, texts, value))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        let mut output = Vec::new();
        for result in results {
            let (id, score, texts, value_bytes) = result
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read row: {}", e)))?;

            // Convert BM25 score to positive
            let positive_score = -score;

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

            // Auto-decode value
            let decoded_value = self.decode_and_deserialize(py, &value_bytes)?;
            output.push((id, positive_score, texts_py, decoded_value));
        }

        Ok(output)
    }

    /// Search with highlighted snippets
    ///
    /// Returns search results with matching text snippets highlighted.
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
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_snippets(
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
        let conn = self.conn.lock();

        // Escape query if requested
        let safe_query = if escape {
            escape_fts5_query(query)
        } else {
            query.to_string()
        };

        if safe_query.is_empty() {
            return Ok(Vec::new());
        }

        // Determine which column to use for snippets
        let col_idx = if let Some(col) = snippet_column {
            self.columns.iter().position(|c| c == col).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Column '{}' not found. Available columns: {:?}",
                    col, &self.columns
                ))
            })?
        } else {
            0 // Default to first column
        };

        let tokens = snippet_tokens.unwrap_or(10);
        let start = highlight_start.unwrap_or("**");
        let end = highlight_end.unwrap_or("**");

        // Use FTS5 snippet() function
        // Note: FTS5 tables require using table name directly for rowid, not aliases
        let sql = format!(
            "SELECT {table}.rowid, bm25({table}), snippet({table}, {col_idx}, '{start}', '{end}', '...', {tokens}), {table}_values.value
             FROM {table}
             JOIN {table}_values ON {table}.value_ref = {table}_values.id
             WHERE {table} MATCH ?
             ORDER BY bm25({table})
             LIMIT ?",
            table = &self.table,
            col_idx = col_idx,
            start = start,
            end = end,
            tokens = tokens
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {}", e)))?;

        let k_i64 = k as i64;
        let results = stmt
            .query_map(params![safe_query, k_i64], |row| {
                let id: i64 = row.get(0)?;
                let score: f64 = row.get(1)?;
                let snippet: String = row.get(2)?;
                let value: Vec<u8> = row.get(3)?;
                Ok((id, score, snippet, value))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        let mut output = Vec::new();
        for result in results {
            let (id, score, snippet, value_bytes) = result
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read row: {}", e)))?;

            // Convert BM25 score to positive
            let positive_score = -score;

            // Auto-decode value
            let decoded_value = self.decode_and_deserialize(py, &value_bytes)?;
            output.push((id, positive_score, snippet, decoded_value));
        }

        Ok(output)
    }

    /// Count documents matching a query
    pub fn count_matches(&self, query: &str) -> PyResult<i64> {
        let conn = self.conn.lock();

        let sql = format!("SELECT COUNT(*) FROM {} WHERE {} MATCH ?", &self.table, &self.table);

        let count: i64 = conn
            .query_row(&sql, params![query], |row| row.get(0))
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        Ok(count)
    }
}
