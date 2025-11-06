//! PyO3 bindings for SkipList

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::core::SkipList;
use crate::tree::pyobject_key::PyObjectKey;
use crate::tree::python::PyValue;

/// Skip List with Python object keys and values
#[pyclass(name = "SkipList")]
pub struct PySkipList {
    inner: SkipList<PyObjectKey, PyValue>,
}

#[pymethods]
impl PySkipList {
    /// Create a new empty SkipList
    #[new]
    fn new() -> Self {
        Self { inner: SkipList::new() }
    }

    /// Insert a key-value pair (lock-free, thread-safe)
    fn insert(&self, key: Py<PyAny>, value: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        let value_wrapper = PyValue::new(value);
        self.inner
            .insert(key_wrapper, value_wrapper)
            .map(|arc_v| Python::with_gil(|py| arc_v.obj.clone_ref(py)))
    }

    /// Get value for a key (lock-free, thread-safe)
    fn get(&self, key: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .get(&key_wrapper)
            .map(|arc_v| Python::with_gil(|py| arc_v.obj.clone_ref(py)))
    }

    /// Remove a key-value pair (lock-free with epoch-based GC)
    fn remove(&self, key: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .remove(&key_wrapper)
            .map(|arc_v| Python::with_gil(|py| arc_v.obj.clone_ref(py)))
    }

    /// Check if key exists
    fn __contains__(&self, key: Py<PyAny>) -> bool {
        let key_wrapper = PyObjectKey::new(key);
        self.inner.get(&key_wrapper).is_some()
    }

    /// Get number of elements
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("SkipList(len={})", self.inner.len())
    }

    /// Dict-like getitem: skiplist[key]
    fn __getitem__(&self, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .get(&key_wrapper)
            .map(|v| Python::with_gil(|py| v.obj.clone_ref(py)))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))
    }

    /// Dict-like setitem: skiplist[key] = value (lock-free, thread-safe)
    fn __setitem__(&self, key: Py<PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let key_wrapper = PyObjectKey::new(key);
        let value_wrapper = PyValue::new(value);
        self.inner.insert(key_wrapper, value_wrapper);
        Ok(())
    }

    /// Dict-like delitem: del skiplist[key] (not supported in lock-free version)
    fn __delitem__(&self, key: Py<PyAny>) -> PyResult<()> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .remove(&key_wrapper)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found"))?;
        Ok(())
    }

    /// Iterate over all key-value pairs in sorted order (snapshot view)
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PySkipListIterator> {
        let py = slf.py();
        let items: Vec<(Py<PyAny>, Py<PyAny>)> = slf
            .inner
            .iter()
            .map(|(k, arc_v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(arc_v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, arc_v.obj.as_ptr())
                        .expect("INCREF failed"),
                )
            })
            .collect();

        Ok(PySkipListIterator { items, index: 0 })
    }

    /// Get all keys as sorted list (snapshot view)
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let keys: Vec<Py<PyAny>> = self
            .inner
            .iter()
            .map(|(k, _)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed")
            })
            .collect();
        Ok(PyList::new_bound(py, keys).into())
    }

    /// Get all values in key order (snapshot view)
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values: Vec<Py<PyAny>> = self
            .inner
            .iter()
            .map(|(_, arc_v)| unsafe {
                ffi::Py_INCREF(arc_v.obj.as_ptr());
                Py::<PyAny>::from_owned_ptr_or_opt(py, arc_v.obj.as_ptr()).expect("INCREF failed")
            })
            .collect();
        Ok(PyList::new_bound(py, values).into())
    }

    /// Get all (key, value) pairs (snapshot view)
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<(Py<PyAny>, Py<PyAny>)> = self
            .inner
            .iter()
            .map(|(k, arc_v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(arc_v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, arc_v.obj.as_ptr())
                        .expect("INCREF failed"),
                )
            })
            .collect();
        Ok(PyList::new_bound(py, items).into())
    }

    /// Clear all entries
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Range query: get all (key, value) pairs where start <= key < end (snapshot view, thread-safe)
    fn range(&self, py: Python<'_>, start: Py<PyAny>, end: Py<PyAny>) -> PyResult<Py<PyList>> {
        let start_wrapper = PyObjectKey::new(start);
        let end_wrapper = PyObjectKey::new(end);

        // Collect from range iterator
        let results: Vec<(Py<PyAny>, Py<PyAny>)> = self
            .inner
            .range(start_wrapper, end_wrapper)
            .map(|(k, arc_v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(arc_v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, arc_v.obj.as_ptr())
                        .expect("INCREF failed"),
                )
            })
            .collect();

        Ok(PyList::new_bound(py, results).into())
    }
}

/// Iterator for SkipList
#[pyclass]
pub struct PySkipListIterator {
    items: Vec<(Py<PyAny>, Py<PyAny>)>,
    index: usize,
}

#[pymethods]
impl PySkipListIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Py<PyAny>, Py<PyAny>)> {
        if slf.index < slf.items.len() {
            let index = slf.index;
            slf.index += 1;
            let (k, v) = &slf.items[index];
            unsafe {
                let py = Python::assume_gil_acquired();
                ffi::Py_INCREF(k.as_ptr());
                ffi::Py_INCREF(v.as_ptr());
                Some((
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, v.as_ptr()).expect("INCREF failed"),
                ))
            }
        } else {
            None
        }
    }
}

/// Register SkipList types with Python module
pub fn register_skiplist_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySkipList>()?;
    m.add_class::<PySkipListIterator>()?;
    Ok(())
}
