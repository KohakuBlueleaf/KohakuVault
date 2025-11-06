//! PyO3 bindings for CSB+Tree with Python object support

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::csbtree::CSBPlusTree;
use super::pyobject_key::PyObjectKey;

/// Wrapper for Py<PyAny> that implements Clone using clone_ref
pub struct PyValue {
    pub(crate) obj: Py<PyAny>,
}

impl PyValue {
    pub(crate) fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }

    pub(crate) fn into_py(self) -> Py<PyAny> {
        self.obj
    }
}

impl Clone for PyValue {
    fn clone(&self) -> Self {
        // OPTIMIZATION: Use unsafe Py_INCREF instead of GIL acquisition
        unsafe {
            ffi::Py_INCREF(self.obj.as_ptr());
            Self {
                obj: Py::from_owned_ptr_or_opt(Python::assume_gil_acquired(), self.obj.as_ptr())
                    .expect("Clone failed"),
            }
        }
    }
}

impl Default for PyValue {
    fn default() -> Self {
        Python::with_gil(|py| Self { obj: py.None() })
    }
}

/// CSB+Tree with Python object keys and values
///
/// This tree uses Python's comparison operators (__lt__, __gt__, __eq__)
/// to order keys, allowing any comparable Python objects to be used.
///
/// Example:
///     tree = CSBTree()
///     tree.insert(10, "ten")
///     tree.insert(5, "five")
///     tree.insert(15, "fifteen")
///
///     # Iterate in sorted order
///     for key, value in tree:
///         print(key, value)  # 5, 10, 15
///
///     # Range query
///     for key, value in tree.range(0, 12):
///         print(key, value)  # 5, 10
#[pyclass(name = "CSBTree")]
pub struct PyCSBTree {
    inner: CSBPlusTree<PyObjectKey, PyValue>,
}

#[pymethods]
impl PyCSBTree {
    /// Create a new empty CSB+Tree
    ///
    /// Args:
    ///     order: Maximum keys per node (default: 63, balanced for most use cases)
    ///            Optimal values based on benchmarks (random insertion):
    ///            - 10K items:   order=15 (fastest: 2.2M ops/sec)
    ///            - 50K items:   order=31 (fastest: 1.8M ops/sec)
    ///            - 100K items:  order=15 (fastest: 1.7M ops/sec)
    ///            - General:     order=63 (good balance)
    ///
    /// Range: 3 to 2047+ (any positive integer)
    #[new]
    #[pyo3(signature = (order=63))]
    fn new(order: usize) -> Self {
        Self { inner: CSBPlusTree::new_with_order(order) }
    }

    /// Insert a key-value pair
    ///
    /// Args:
    ///     key: Any comparable Python object (must support <, >, ==)
    ///     value: Any Python object
    ///
    /// Returns:
    ///     The old value if key existed, None otherwise
    fn insert(&mut self, key: Py<PyAny>, value: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        let value_wrapper = PyValue::new(value);
        self.inner
            .insert(key_wrapper, value_wrapper)
            .map(|v| v.into_py())
    }

    /// Get value for a key
    ///
    /// Args:
    ///     key: Key to look up
    ///
    /// Returns:
    ///     Value if key exists, None otherwise
    fn get(&self, key: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner.get(&key_wrapper).cloned().map(|v| v.into_py())
    }

    /// Remove a key-value pair
    ///
    /// Args:
    ///     key: Key to remove
    ///
    /// Returns:
    ///     The old value if key existed, None otherwise
    fn remove(&mut self, key: Py<PyAny>) -> Option<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner.remove(&key_wrapper).map(|v| v.into_py())
    }

    /// Check if key exists in tree
    ///
    /// Args:
    ///     key: Key to check
    ///
    /// Returns:
    ///     True if key exists, False otherwise
    fn __contains__(&self, key: Py<PyAny>) -> bool {
        let key_wrapper = PyObjectKey::new(key);
        self.inner.get(&key_wrapper).is_some()
    }

    /// Get number of key-value pairs
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("CSBTree(len={})", self.inner.len())
    }

    /// Dict-like getitem: tree[key]
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .get(&key_wrapper)
            .map(|v| v.obj.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found in CSBTree"))
    }

    /// Dict-like setitem: tree[key] = value
    fn __setitem__(&mut self, key: Py<PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let key_wrapper = PyObjectKey::new(key);
        let value_wrapper = PyValue::new(value);
        self.inner.insert(key_wrapper, value_wrapper);
        Ok(())
    }

    /// Dict-like delitem: del tree[key]
    fn __delitem__(&mut self, key: Py<PyAny>) -> PyResult<()> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .remove(&key_wrapper)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Key not found in CSBTree"))?;
        Ok(())
    }

    /// Iterate over all key-value pairs in sorted order
    ///
    /// Yields:
    ///     (key, value) tuples in ascending key order
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyCSBTreeIterator> {
        let py = slf.py();
        // OPTIMIZATION: Use unsafe INCREF to avoid clone_ref overhead
        let items: Vec<(Py<PyAny>, Py<PyAny>)> = slf
            .inner
            .iter()
            .map(|(k, v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, v.obj.as_ptr()).expect("INCREF failed"),
                )
            })
            .collect();

        Ok(PyCSBTreeIterator { items, index: 0 })
    }

    /// Get all keys as sorted list
    ///
    /// Returns:
    ///     List of keys in ascending order
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        // OPTIMIZATION: Use unsafe INCREF to avoid clone_ref overhead
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

    /// Get all values in key order
    ///
    /// Returns:
    ///     List of values in ascending key order
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        // OPTIMIZATION: Use unsafe INCREF to avoid clone_ref overhead
        let values: Vec<Py<PyAny>> = self
            .inner
            .iter()
            .map(|(_, v)| unsafe {
                ffi::Py_INCREF(v.obj.as_ptr());
                Py::<PyAny>::from_owned_ptr_or_opt(py, v.obj.as_ptr()).expect("INCREF failed")
            })
            .collect();
        Ok(PyList::new_bound(py, values).into())
    }

    /// Get all (key, value) pairs
    ///
    /// Returns:
    ///     List of (key, value) tuples in ascending key order
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        // OPTIMIZATION: Use unsafe INCREF to avoid clone_ref overhead
        let items: Vec<(Py<PyAny>, Py<PyAny>)> = self
            .inner
            .iter()
            .map(|(k, v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, v.obj.as_ptr()).expect("INCREF failed"),
                )
            })
            .collect();
        Ok(PyList::new_bound(py, items).into())
    }

    /// Range query: get all (key, value) pairs where start <= key < end
    ///
    /// Args:
    ///     start: Start of range (inclusive)
    ///     end: End of range (exclusive)
    ///
    /// Returns:
    ///     List of (key, value) tuples in range
    fn range(&self, py: Python<'_>, start: Py<PyAny>, end: Py<PyAny>) -> PyResult<Py<PyList>> {
        let start_wrapper = PyObjectKey::new(start);
        let end_wrapper = PyObjectKey::new(end);

        // OPTIMIZATION: Use unsafe INCREF to avoid clone_ref overhead
        let results: Vec<(Py<PyAny>, Py<PyAny>)> = self
            .inner
            .range(start_wrapper..end_wrapper)
            .map(|(k, v)| unsafe {
                ffi::Py_INCREF(k.obj.as_ptr());
                ffi::Py_INCREF(v.obj.as_ptr());
                (
                    Py::<PyAny>::from_owned_ptr_or_opt(py, k.obj.as_ptr()).expect("INCREF failed"),
                    Py::<PyAny>::from_owned_ptr_or_opt(py, v.obj.as_ptr()).expect("INCREF failed"),
                )
            })
            .collect();

        Ok(PyList::new_bound(py, results).into())
    }

    /// Clear all entries
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Check if tree is empty
    ///
    /// Returns:
    ///     True if tree has no entries
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get value with default if key doesn't exist
    ///
    /// Args:
    ///     key: Key to look up
    ///     default: Value to return if key not found
    ///
    /// Returns:
    ///     Value if key exists, default otherwise
    fn get_default(&self, py: Python<'_>, key: Py<PyAny>, default: Py<PyAny>) -> Py<PyAny> {
        let key_wrapper = PyObjectKey::new(key);
        self.inner
            .get(&key_wrapper)
            .map(|v| v.obj.clone_ref(py))
            .unwrap_or(default)
    }

    /// Insert key-value pair, return value (for dict-like setdefault)
    ///
    /// Args:
    ///     key: Key to insert
    ///     value: Value to insert if key doesn't exist
    ///
    /// Returns:
    ///     Existing value if key existed, new value otherwise
    fn setdefault(&mut self, py: Python<'_>, key: Py<PyAny>, value: Py<PyAny>) -> Py<PyAny> {
        let key_wrapper = PyObjectKey::new(key);

        if let Some(existing) = self.inner.get(&key_wrapper) {
            existing.obj.clone_ref(py)
        } else {
            let value_wrapper = PyValue::new(value.clone_ref(py));
            self.inner.insert(key_wrapper, value_wrapper);
            value
        }
    }

    /// Update tree with items from another dict/tree
    ///
    /// Args:
    ///     items: Iterable of (key, value) tuples
    fn update(&mut self, py: Python<'_>, items: Py<PyAny>) -> PyResult<()> {
        let items_bound = items.bind(py);

        for item in items_bound.iter()? {
            let tuple = item?;
            let key = tuple.get_item(0)?;
            let value = tuple.get_item(1)?;

            let key_wrapper = PyObjectKey::new(key.unbind());
            let value_wrapper = PyValue::new(value.unbind());
            self.inner.insert(key_wrapper, value_wrapper);
        }

        Ok(())
    }
}

/// Iterator for CSB+Tree
#[pyclass]
pub struct PyCSBTreeIterator {
    items: Vec<(Py<PyAny>, Py<PyAny>)>,
    index: usize,
}

#[pymethods]
impl PyCSBTreeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Py<PyAny>, Py<PyAny>)> {
        if slf.index < slf.items.len() {
            let index = slf.index;
            slf.index += 1;
            let (k, v) = &slf.items[index];

            // OPTIMIZATION: Use unsafe INCREF
            unsafe {
                let py = Python::assume_gil_acquired();
                ffi::Py_INCREF(k.as_ptr());
                ffi::Py_INCREF(v.as_ptr());
                Some((
                    Py::from_owned_ptr_or_opt(py, k.as_ptr()).expect("INCREF failed"),
                    Py::from_owned_ptr_or_opt(py, v.as_ptr()).expect("INCREF failed"),
                ))
            }
        } else {
            None
        }
    }
}

/// Register CSB+Tree types with Python module
pub fn register_tree_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCSBTree>()?;
    m.add_class::<PyCSBTreeIterator>()?;
    Ok(())
}
