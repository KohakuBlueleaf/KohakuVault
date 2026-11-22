//! Pure-Rust TextVault client over SQLite FTS5 + companion blob table.
//!
//! Uses the exact same `{table}` (FTS5 virtual table with `value_ref UNINDEXED`)
//! and `{table}_values` SQLite schema as the Python `_TextVault` binding.

use std::time::Duration;

use parking_lot::Mutex;
use rusqlite::{
    params, params_from_iter, Connection, OpenFlags, OptionalExtension, TransactionBehavior,
};

use crate::common::VaultError;

/// One BM25-ranked search result from [`TextVault::search`].
#[derive(Debug, Clone)]
pub struct TextSearchHit {
    pub doc_id: i64,
    pub score: f64,
    pub texts: Vec<String>,
    pub value: Vec<u8>,
}

/// One sequential document row from [`TextVault::tail`].
#[derive(Debug, Clone)]
pub struct TextRecord {
    pub doc_id: i64,
    pub texts: Vec<String>,
    pub value: Vec<u8>,
}

fn validate_identifier(name: &str) -> Result<(), VaultError> {
    if name.is_empty() || name.chars().any(|c| !c.is_alphanumeric() && c != '_') {
        return Err(VaultError::Invalid(format!(
            "invalid SQL identifier '{name}': only ASCII alphanumeric and '_' are allowed"
        )));
    }
    Ok(())
}

/// Escape an arbitrary string for literal FTS5 phrase matching.
pub fn escape_fts5_query(query: &str) -> String {
    format!("\"{}\"", query.replace('"', "\"\""))
}

/// Pure-Rust SQLite FTS5 full-text store compatible with Python `kohakuvault.TextVault`.
pub struct TextVault {
    conn: Mutex<Option<Connection>>,
    table: String,
    columns: Vec<String>,
    read_only: bool,
}

impl TextVault {
    /// Open or create a TextVault at `path` (or `":memory:"`).
    pub fn open(
        path: &str,
        table: &str,
        columns: Option<Vec<String>>,
        read_only: bool,
    ) -> Result<Self, VaultError> {
        validate_identifier(table)?;
        let cols = columns.unwrap_or_else(|| vec!["content".to_string()]);
        if cols.is_empty() {
            return Err(VaultError::Invalid("TextVault requires at least one column".into()));
        }
        for col in &cols {
            validate_identifier(col)?;
        }

        let flags = if read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_URI
        };
        let conn = Connection::open_with_flags(path, flags)?;
        conn.busy_timeout(Duration::from_secs(5))?;

        if !read_only {
            if path != ":memory:" {
                let _ = conn.pragma_update(None, "journal_mode", "WAL");
            }
            let _ = conn.pragma_update(None, "synchronous", "NORMAL");
            let _ = conn.pragma_update(None, "temp_store", "MEMORY");

            let indexed_cols = cols.join(", ");
            conn.execute(
                &format!(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5({indexed_cols}, value_ref UNINDEXED)"
                ),
                [],
            )?;
            conn.execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {table}_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        value BLOB NOT NULL
                    )"
                ),
                [],
            )?;
        }

        Ok(Self {
            conn: Mutex::new(Some(conn)),
            table: table.to_string(),
            columns: cols,
            read_only,
        })
    }

    /// Insert one document and return its FTS5 `rowid`.
    pub fn insert(&self, texts: &[&str], value: &[u8]) -> Result<i64, VaultError> {
        if self.read_only {
            return Err(VaultError::ReadOnly(format!("TextVault '{}' is read-only", self.table)));
        }
        if texts.len() != self.columns.len() {
            return Err(VaultError::Invalid(format!(
                "expected {} text columns, got {}",
                self.columns.len(),
                texts.len()
            )));
        }
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value_id: i64 = tx.query_row(
            &format!("INSERT INTO {}_values (value) VALUES (?) RETURNING id", self.table),
            params![value],
            |row| row.get(0),
        )?;
        let placeholders = vec!["?"; self.columns.len()].join(", ");
        let doc_sql = format!(
            "INSERT INTO {} ({}, value_ref) VALUES ({}, ?)",
            self.table,
            self.columns.join(", "),
            placeholders
        );
        let mut param_values: Vec<rusqlite::types::Value> = texts
            .iter()
            .map(|s| rusqlite::types::Value::Text((*s).to_string()))
            .collect();
        param_values.push(rusqlite::types::Value::Integer(value_id));
        tx.execute(&doc_sql, params_from_iter(param_values))?;
        let doc_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(doc_id)
    }

    /// Insert many `(texts, value)` documents atomically in one transaction.
    pub fn insert_many(&self, items: &[(Vec<String>, Vec<u8>)]) -> Result<Vec<i64>, VaultError> {
        if self.read_only {
            return Err(VaultError::ReadOnly(format!("TextVault '{}' is read-only", self.table)));
        }
        if items.is_empty() {
            return Ok(Vec::new());
        }
        for (texts, _) in items {
            if texts.len() != self.columns.len() {
                return Err(VaultError::Invalid(format!(
                    "expected {} text columns, got {}",
                    self.columns.len(),
                    texts.len()
                )));
            }
        }
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let placeholders = vec!["?"; self.columns.len()].join(", ");
        let doc_sql = format!(
            "INSERT INTO {} ({}, value_ref) VALUES ({}, ?)",
            self.table,
            self.columns.join(", "),
            placeholders
        );
        let value_sql =
            format!("INSERT INTO {}_values (value) VALUES (?) RETURNING id", self.table);
        let mut ids = Vec::with_capacity(items.len());
        {
            let mut val_stmt = tx.prepare(&value_sql)?;
            let mut doc_stmt = tx.prepare(&doc_sql)?;
            for (texts, value) in items {
                let value_id: i64 = val_stmt.query_row(params![value], |r| r.get(0))?;
                let mut params_vec: Vec<rusqlite::types::Value> = texts
                    .iter()
                    .cloned()
                    .map(rusqlite::types::Value::Text)
                    .collect();
                params_vec.push(rusqlite::types::Value::Integer(value_id));
                doc_stmt.execute(params_from_iter(params_vec))?;
                ids.push(tx.last_insert_rowid());
            }
        }
        tx.commit()?;
        Ok(ids)
    }

    /// Search via FTS5 BM25 ranking (higher score = more relevant).
    pub fn search(
        &self,
        query: &str,
        k: usize,
        column: Option<&str>,
        escape: bool,
    ) -> Result<Vec<TextSearchHit>, VaultError> {
        if k == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = if escape {
            escape_fts5_query(query)
        } else {
            query.to_string()
        };
        let match_expr = match column {
            Some(col) => {
                validate_identifier(col)?;
                format!("{col} : {fts_query}")
            }
            None => fts_query,
        };
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        let text_cols = self
            .columns
            .iter()
            .map(|c| format!("f.{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT f.rowid, -bm25({t}) AS score, v.value, {text_cols}
             FROM {t} f
             JOIN {t}_values v ON f.value_ref = v.id
             WHERE {t} MATCH ?1
             ORDER BY bm25({t})
             LIMIT ?2",
            t = self.table
        );
        let mut stmt = conn.prepare(&sql)?;
        let col_count = self.columns.len();
        let rows = stmt.query_map(params![match_expr, k as i64], |row| {
            let doc_id: i64 = row.get(0)?;
            let score: f64 = row.get(1)?;
            let value: Vec<u8> = row.get(2)?;
            let mut texts = Vec::with_capacity(col_count);
            for i in 0..col_count {
                texts.push(row.get::<_, String>(3 + i)?);
            }
            Ok(TextSearchHit { doc_id, score, texts, value })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Return newest-first records strictly before `before_id` (or from the end when `None`).
    pub fn tail(
        &self,
        before_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TextRecord>, VaultError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        let text_cols = self
            .columns
            .iter()
            .map(|c| format!("f.{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        let col_count = self.columns.len();
        let sql = match before_id {
            Some(_) => format!(
                "SELECT f.rowid, v.value, {text_cols}
                 FROM {t} f
                 JOIN {t}_values v ON f.value_ref = v.id
                 WHERE f.rowid < ?1
                 ORDER BY f.rowid DESC
                 LIMIT ?2",
                t = self.table
            ),
            None => format!(
                "SELECT f.rowid, v.value, {text_cols}
                 FROM {t} f
                 JOIN {t}_values v ON f.value_ref = v.id
                 ORDER BY f.rowid DESC
                 LIMIT ?1",
                t = self.table
            ),
        };
        let mut stmt = conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<TextRecord> {
            let doc_id: i64 = row.get(0)?;
            let value: Vec<u8> = row.get(1)?;
            let mut texts = Vec::with_capacity(col_count);
            for i in 0..col_count {
                texts.push(row.get::<_, String>(2 + i)?);
            }
            Ok(TextRecord { doc_id, texts, value })
        };
        let rows = match before_id {
            Some(cursor) => stmt.query_map(params![cursor, limit as i64], map_row)?,
            None => stmt.query_map(params![limit as i64], map_row)?,
        };
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Fetch stored value bytes by `doc_id`.
    pub fn get(&self, doc_id: i64) -> Result<Option<Vec<u8>>, VaultError> {
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        let sql = format!(
            "SELECT v.value FROM {t} f JOIN {t}_values v ON f.value_ref = v.id WHERE f.rowid = ?1",
            t = self.table
        );
        Ok(conn
            .query_row(&sql, params![doc_id], |r| r.get(0))
            .optional()?)
    }

    /// Count indexed documents.
    pub fn count(&self) -> Result<i64, VaultError> {
        let guard = self.conn.lock();
        let conn = guard.as_ref().ok_or(VaultError::Closed)?;
        Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {}", self.table), [], |r| r.get(0))?)
    }

    /// Delete document by `doc_id`.
    pub fn delete(&self, doc_id: i64) -> Result<bool, VaultError> {
        if self.read_only {
            return Err(VaultError::ReadOnly(format!("TextVault '{}' is read-only", self.table)));
        }
        let mut guard = self.conn.lock();
        let conn = guard.as_mut().ok_or(VaultError::Closed)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value_ref: Option<i64> = tx
            .query_row(
                &format!("SELECT value_ref FROM {} WHERE rowid = ?1", self.table),
                params![doc_id],
                |r| r.get(0),
            )
            .optional()?;
        let Some(vref) = value_ref else {
            return Ok(false);
        };
        tx.execute(&format!("DELETE FROM {} WHERE rowid = ?1", self.table), params![doc_id])?;
        tx.execute(&format!("DELETE FROM {}_values WHERE id = ?1", self.table), params![vref])?;
        tx.commit()?;
        Ok(true)
    }

    /// Close the underlying SQLite connection.
    pub fn close(&self) {
        let _ = self.conn.lock().take();
    }
}

#[cfg(test)]
mod tests {
    use super::TextVault;

    #[test]
    fn insert_search_and_tail_roundtrip() {
        let tv = TextVault::open(":memory:", "kb_logs", None, false).unwrap();
        let ids = tv
            .insert_many(&[
                (vec!["epoch 1 loss=1.23".into()], b"meta1".to_vec()),
                (vec!["step 400 NaN loss detected".into()], b"meta2".to_vec()),
                (vec!["checkpoint saved ckpt-500".into()], b"meta3".to_vec()),
            ])
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(tv.count().unwrap(), 3);

        let hits = tv.search("NaN", 5, None, true).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, ids[1]);
        assert_eq!(hits[0].value, b"meta2");

        let page1 = tv.tail(None, 2).unwrap();
        assert_eq!(page1.len(), 2);
        assert_eq!(page1[0].doc_id, ids[2]);
        assert_eq!(page1[1].doc_id, ids[1]);

        let page2 = tv.tail(Some(page1[1].doc_id), 2).unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].doc_id, ids[0]);
        tv.close();
    }
}
