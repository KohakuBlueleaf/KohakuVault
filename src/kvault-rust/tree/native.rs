//! Native type CSBTree variants for better performance
//!
//! These avoid Python comparison overhead by using native Rust types

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use super::csbtree::CSBPlusTree;

/// CSBTree with i64 keys and bytes values (FAST - no Python comparison overhead!)
#[pyclass(name = "CSBTreeI64")]
pub struct CSBTreeI64 {
    inner: CSBPlusTree<i64, Vec<u8>>,
}

#[pymethods]
impl CSBTreeI64 {
    #[new]
    #[pyo3(signature = (order=63))]
    fn new(order: usize) -> Self {
        Self {
            inner: CSBPlusTree::new_with_order(order),
        }
    }

    fn insert(&mut self, py: Python<'_>, key: i64, value: &Bound<'_, PyBytes>) -> Option<Py<PyBytes>> {
        let v = value.as_bytes().to_vec();
        self.inner.insert(key, v).map(|old| PyBytes::new_bound(py, &old).into())
    }

    fn get(&self, py: Python<'_>, key: i64) -> Option<Py<PyBytes>> {
        self.inner.get(&key).map(|v| PyBytes::new_bound(py, v).into())
    }

    fn remove(&mut self, py: Python<'_>, key: i64) -> Option<Py<PyBytes>> {
        self.inner.remove(&key).map(|v| PyBytes::new_bound(py, &v).into())
    }

    fn __contains__(&self, key: i64) -> bool {
        self.inner.get(&key).is_some()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("CSBTreeI64(len={}, order={})", self.inner.len(), self.inner.max_keys)
    }

    fn __getitem__(&self, py: Python<'_>, key: i64) -> PyResult<Py<PyBytes>> {
        self.inner
            .get(&key)
            .map(|v| PyBytes::new_bound(py, v).into())
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))
    }

    fn __setitem__(&mut self, key: i64, value: &Bound<'_, PyBytes>) -> PyResult<()> {
        let v = value.as_bytes().to_vec();
        self.inner.insert(key, v);
        Ok(())
    }

    fn __delitem__(&mut self, key: i64) -> PyResult<()> {
        self.inner
            .remove(&key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))?;
        Ok(())
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let keys: Vec<i64> = self.inner.iter().map(|(k, _)| *k).collect();
        Ok(PyList::new_bound(py, keys).into())
    }

    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values: Vec<_> = self.inner.iter().map(|(_, v)| PyBytes::new_bound(py, v)).collect();
        Ok(PyList::new_bound(py, values).into())
    }

    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<_> = self
            .inner
            .iter()
            .map(|(k, v)| (*k, PyBytes::new_bound(py, v)))
            .collect();
        Ok(PyList::new_bound(py, items).into())
    }

    fn range(&self, py: Python<'_>, start: i64, end: i64) -> PyResult<Py<PyList>> {
        let results: Vec<_> = self
            .inner
            .range(start..end)
            .map(|(k, v)| (*k, PyBytes::new_bound(py, v)))
            .collect();
        Ok(PyList::new_bound(py, results).into())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let items: Vec<(i64, Py<PyBytes>)> = slf
            .inner
            .iter()
            .map(|(k, v)| (*k, PyBytes::new_bound(py, v).into()))
            .collect();
        Ok(items.into_py(py))
    }
}

/// CSBTree with String keys and bytes values
#[pyclass(name = "CSBTreeString")]
pub struct CSBTreeString {
    inner: CSBPlusTree<String, Vec<u8>>,
}

#[pymethods]
impl CSBTreeString {
    #[new]
    #[pyo3(signature = (order=63))]
    fn new(order: usize) -> Self {
        Self {
            inner: CSBPlusTree::new_with_order(order),
        }
    }

    fn insert(&mut self, py: Python<'_>, key: String, value: &Bound<'_, PyBytes>) -> Option<Py<PyBytes>> {
        let v = value.as_bytes().to_vec();
        self.inner.insert(key, v).map(|old| PyBytes::new_bound(py, &old).into())
    }

    fn get(&self, py: Python<'_>, key: String) -> Option<Py<PyBytes>> {
        self.inner.get(&key).map(|v| PyBytes::new_bound(py, v).into())
    }

    fn remove(&mut self, py: Python<'_>, key: String) -> Option<Py<PyBytes>> {
        self.inner.remove(&key).map(|v| PyBytes::new_bound(py, &v).into())
    }

    fn __contains__(&self, key: String) -> bool {
        self.inner.get(&key).is_some()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("CSBTreeString(len={})", self.inner.len())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyBytes>> {
        self.inner
            .get(&key)
            .map(|v| PyBytes::new_bound(py, v).into())
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))
    }

    fn __setitem__(&mut self, key: String, value: &Bound<'_, PyBytes>) -> PyResult<()> {
        let v = value.as_bytes().to_vec();
        self.inner.insert(key, v);
        Ok(())
    }

    fn __delitem__(&mut self, key: String) -> PyResult<()> {
        self.inner
            .remove(&key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))?;
        Ok(())
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let keys: Vec<String> = self.inner.iter().map(|(k, _)| k.clone()).collect();
        Ok(PyList::new_bound(py, keys).into())
    }

    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values: Vec<_> = self.inner.iter().map(|(_, v)| PyBytes::new_bound(py, v)).collect();
        Ok(PyList::new_bound(py, values).into())
    }

    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<_> = self
            .inner
            .iter()
            .map(|(k, v)| (k.clone(), PyBytes::new_bound(py, v)))
            .collect();
        Ok(PyList::new_bound(py, items).into())
    }

    fn range(&self, py: Python<'_>, start: String, end: String) -> PyResult<Py<PyList>> {
        let results: Vec<_> = self
            .inner
            .range(start..end)
            .map(|(k, v)| (k.clone(), PyBytes::new_bound(py, v)))
            .collect();
        Ok(PyList::new_bound(py, results).into())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let items: Vec<(String, Py<PyBytes>)> = slf
            .inner
            .iter()
            .map(|(k, v)| (k.clone(), PyBytes::new_bound(py, v).into()))
            .collect();
        Ok(items.into_py(py))
    }
}

/// Register native CSBTree types
pub fn register_native_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CSBTreeI64>()?;
    m.add_class::<CSBTreeString>()?;
    Ok(())
}
