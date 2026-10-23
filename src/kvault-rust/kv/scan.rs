//! Keyset pagination for complete, bounded-memory KV iteration.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params_from_iter, types::Value};

use super::ops::to_key_bytes;
use super::{_KVault, cache::SharedValue};
use crate::common::VaultError;

type RawItems = Vec<(Vec<u8>, SharedValue)>;

fn prefix_end(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(byte) = end.pop() {
        if byte != 255 {
            end.push(byte + 1);
            return Some(end);
        }
    }
    None
}

impl _KVault {
    fn scan_page(
        &self,
        prefix: Option<Vec<u8>>,
        limit: Option<usize>,
        after: Option<Vec<u8>>,
        include_values: bool,
    ) -> Result<RawItems, VaultError> {
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let sql_limit = limit
            .map(i64::try_from)
            .transpose()
            .map_err(|_| VaultError::Py("iteration limit is too large".into()))?
            .unwrap_or(-1);
        let fields = if include_values {
            "key, value"
        } else {
            "key, X''"
        };
        let mut sql = format!("SELECT {fields} FROM {} WHERE 1 = 1", self.table);
        let mut parameters = Vec::<Value>::new();

        if let Some(prefix) = &prefix {
            sql.push_str(" AND key >= ?");
            parameters.push(Value::Blob(prefix.clone()));
            if let Some(end) = prefix_end(prefix) {
                sql.push_str(" AND key < ?");
                parameters.push(Value::Blob(end));
            }
        }
        if let Some(after) = &after {
            sql.push_str(" AND key > ?");
            parameters.push(Value::Blob(after.clone()));
        }
        sql.push_str(" ORDER BY key LIMIT ?");
        parameters.push(Value::Integer(sql_limit));

        // Snapshot only the requested overlay page, with shared immutable payloads.
        let pending = {
            let guard = self.cache.lock().unwrap();
            guard
                .as_ref()
                .map(|cache| cache.page(prefix.as_deref(), after.as_deref(), limit, include_values))
                .unwrap_or_default()
        };

        let mut rows = self.with_reader(|conn| {
            let mut statement = conn.prepare_cached(&sql)?;
            let result = statement.query_map(params_from_iter(parameters), |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, Arc::new(row.get::<_, Vec<u8>>(1)?)))
            })?;
            Ok(result.collect::<Result<BTreeMap<_, _>, _>>()?)
        })?;
        rows.extend(pending);
        Ok(rows.into_iter().take(limit.unwrap_or(usize::MAX)).collect())
    }

    pub(crate) fn scan_keys_page(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        after: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let prefix = prefix.map(|value| to_key_bytes(py, value)).transpose()?;
        let after = after.map(|value| to_key_bytes(py, value)).transpose()?;
        let rows = py.allow_threads(|| self.scan_page(prefix, limit, after, false))?;
        Ok(rows
            .into_iter()
            .map(|(key, _)| PyBytes::new(py, &key).unbind())
            .collect())
    }

    pub(crate) fn scan_items_page(
        &self,
        py: Python<'_>,
        prefix: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        after: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(Py<PyBytes>, PyObject)>> {
        let prefix = prefix.map(|value| to_key_bytes(py, value)).transpose()?;
        let after = after.map(|value| to_key_bytes(py, value)).transpose()?;
        let rows = py.allow_threads(|| self.scan_page(prefix, limit, after, true))?;
        rows.into_iter()
            .map(|(key, value)| {
                Ok((PyBytes::new(py, &key).unbind(), self.decode_result(py, &value)?))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::prefix_end;

    #[test]
    fn arbitrary_binary_prefix_bounds() {
        assert_eq!(prefix_end(b"ab"), Some(b"ac".to_vec()));
        assert_eq!(prefix_end(b"a\xff"), Some(b"b".to_vec()));
        assert_eq!(prefix_end(b"\xff\xff"), None);
        assert_eq!(prefix_end(b""), None);
    }
}
