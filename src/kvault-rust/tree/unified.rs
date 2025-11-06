//! Unified CSBTree class with configurable key and value types

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use super::csbtree::CSBPlusTree;
use super::pyobject_key::PyObjectKey;
use super::python::PyValue;

/// Internal tree storage enum
enum TreeStorage {
    PyObjectPyObject(CSBPlusTree<PyObjectKey, PyValue>),
    PyObjectBytes(CSBPlusTree<PyObjectKey, Vec<u8>>),
    I64PyObject(CSBPlusTree<i64, PyValue>),
    I64Bytes(CSBPlusTree<i64, Vec<u8>>),
    F64PyObject(CSBPlusTree<f64, PyValue>),
    F64Bytes(CSBPlusTree<f64, Vec<u8>>),
    StringPyObject(CSBPlusTree<String, PyValue>),
    StringBytes(CSBPlusTree<String, Vec<u8>>),
}

/// Unified CSBTree with configurable key and value types
#[pyclass(name = "CSBTree")]
pub struct UnifiedCSBTree {
    storage: TreeStorage,
    key_type: String,
    value_type: String,
}

#[pymethods]
impl UnifiedCSBTree {
    /// Create a new CSBTree
    ///
    /// Args:
    ///     key_type: Type of keys - "pyobject" (default), "i64", "f64", or "text"
    ///     value_type: Type of values - "pyobject" (default) or "bytes"
    ///     order: Max keys per node (default: 63, recommended: 15-31 for most cases)
    ///
    /// Examples:
    ///     tree = CSBTree()  # PyObject keys & values (flexible)
    ///     tree = CSBTree(key_type="i64", value_type="bytes")  # Fast integer keys
    ///     tree = CSBTree(key_type="text", value_type="pyobject")  # String keys, any values
    #[new]
    #[pyo3(signature = (key_type="pyobject", value_type="pyobject", order=63))]
    fn new(key_type: &str, value_type: &str, order: usize) -> PyResult<Self> {
        let storage = match (key_type, value_type) {
            ("pyobject", "pyobject") => TreeStorage::PyObjectPyObject(CSBPlusTree::new_with_order(order)),
            ("pyobject", "bytes") => TreeStorage::PyObjectBytes(CSBPlusTree::new_with_order(order)),
            ("i64", "pyobject") => TreeStorage::I64PyObject(CSBPlusTree::new_with_order(order)),
            ("i64", "bytes") => TreeStorage::I64Bytes(CSBPlusTree::new_with_order(order)),
            ("f64", "pyobject") => TreeStorage::F64PyObject(CSBPlusTree::new_with_order(order)),
            ("f64", "bytes") => TreeStorage::F64Bytes(CSBPlusTree::new_with_order(order)),
            ("text", "pyobject") => TreeStorage::StringPyObject(CSBPlusTree::new_with_order(order)),
            ("text", "bytes") => TreeStorage::StringBytes(CSBPlusTree::new_with_order(order)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid key_type '{}' or value_type '{}'. \
                        key_type must be: pyobject, i64, f64, text. \
                        value_type must be: pyobject, bytes", key_type, value_type)
            )),
        };

        Ok(Self {
            storage,
            key_type: key_type.to_string(),
            value_type: value_type.to_string(),
        })
    }

    fn __repr__(&self) -> String {
        let len = match &self.storage {
            TreeStorage::PyObjectPyObject(t) => t.len(),
            TreeStorage::PyObjectBytes(t) => t.len(),
            TreeStorage::I64PyObject(t) => t.len(),
            TreeStorage::I64Bytes(t) => t.len(),
            TreeStorage::F64PyObject(t) => t.len(),
            TreeStorage::F64Bytes(t) => t.len(),
            TreeStorage::StringPyObject(t) => t.len(),
            TreeStorage::StringBytes(t) => t.len(),
        };
        format!("CSBTree(key_type='{}', value_type='{}', len={})",
                self.key_type, self.value_type, len)
    }

    fn __len__(&self) -> usize {
        match &self.storage {
            TreeStorage::PyObjectPyObject(t) => t.len(),
            TreeStorage::PyObjectBytes(t) => t.len(),
            TreeStorage::I64PyObject(t) => t.len(),
            TreeStorage::I64Bytes(t) => t.len(),
            TreeStorage::F64PyObject(t) => t.len(),
            TreeStorage::F64Bytes(t) => t.len(),
            TreeStorage::StringPyObject(t) => t.len(),
            TreeStorage::StringBytes(t) => t.len(),
        }
    }

    // TODO: Implement all methods with match on storage...
    // This is getting very verbose. Let me use macros or a different approach.
}

pub fn register_unified_tree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UnifiedCSBTree>()?;
    Ok(())
}
