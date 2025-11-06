//! Wrapper for Python objects as keys with comparison support

use std::cmp::Ordering;

use pyo3::ffi;
use pyo3::prelude::*;

/// Wrapper for Python objects that implements Ord using Python's comparison protocol
pub struct PyObjectKey {
    pub obj: Py<PyAny>,
}

impl PyObjectKey {
    pub(crate) fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

impl Clone for PyObjectKey {
    fn clone(&self) -> Self {
        // OPTIMIZATION: Use unsafe Py_INCREF instead of GIL acquisition
        // This is safe because we're just incrementing the reference count
        unsafe {
            ffi::Py_INCREF(self.obj.as_ptr());
            Self {
                obj: Py::from_owned_ptr_or_opt(Python::assume_gil_acquired(), self.obj.as_ptr())
                    .expect("Clone failed"),
            }
        }
    }
}

impl PartialEq for PyObjectKey {
    fn eq(&self, other: &Self) -> bool {
        // OPTIMIZATION: Use FFI directly without GIL re-acquisition
        unsafe {
            let self_ptr = self.obj.as_ptr();
            let other_ptr = other.obj.as_ptr();

            // Try __eq__ (Py_EQ = 2)
            let eq_result = ffi::PyObject_RichCompareBool(self_ptr, other_ptr, ffi::Py_EQ);
            eq_result == 1
        }
    }
}

impl Eq for PyObjectKey {}

impl Default for PyObjectKey {
    fn default() -> Self {
        Python::with_gil(|py| Self { obj: py.None() })
    }
}

impl PartialOrd for PyObjectKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PyObjectKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // OPTIMIZATION: Use FFI directly without GIL re-acquisition
        // This is safe because:
        // 1. We're called from Python code which already holds GIL
        // 2. PyObject_RichCompareBool is thread-safe when GIL is held
        // 3. We don't release GIL during comparison
        unsafe {
            let self_ptr = self.obj.as_ptr();
            let other_ptr = other.obj.as_ptr();

            // Try __lt__ (Py_LT = 0)
            let lt_result = ffi::PyObject_RichCompareBool(self_ptr, other_ptr, ffi::Py_LT);
            if lt_result == 1 {
                return Ordering::Less;
            }

            // Try __gt__ (Py_GT = 4)
            let gt_result = ffi::PyObject_RichCompareBool(self_ptr, other_ptr, ffi::Py_GT);
            if gt_result == 1 {
                return Ordering::Greater;
            }

            // Default to Equal
            Ordering::Equal
        }
    }
}
