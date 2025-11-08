//! Optimized bulk pack/unpack operations for vectors
//!
//! Provides significant performance improvements by batching numpy operations
//! and minimizing Python/Rust boundary crossings.

use super::vector::ElementType;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// Optimized bulk pack for fixed-shape vectors
///
/// Instead of calling pack for each vector, we:
/// 1. Stack all arrays into one big numpy array
/// 2. Get bytes once
/// 3. Interleave type bytes efficiently
pub fn pack_many_vectors_fixed(
    py: Python,
    values: &Bound<PyList>,
    element_type: ElementType,
    shape: &[usize],
) -> Result<Py<PyBytes>, PyErr> {
    let count = values.len();

    if count == 0 {
        return Ok(PyBytes::new_bound(py, &[]).unbind());
    }

    // Import numpy
    let numpy = py.import_bound("numpy")?;
    let dtype_str = element_type.to_numpy_dtype();

    // Calculate sizes
    let elem_count: usize = shape.iter().product();
    let data_bytes_per_vec = elem_count * element_type.byte_size();
    let total_size = count * (1 + data_bytes_per_vec); // type(1) + data per vector

    let mut result = Vec::with_capacity(total_size);
    let type_byte = element_type as u8;

    // Optimization: Try to stack all arrays and get bytes once
    // This is much faster than processing each array individually
    let stacked_result: Result<Bound<PyAny>, PyErr> = (|| {
        // Stack all arrays into one big array
        let stacked = numpy.call_method1("stack", (values,))?;
        // Ensure correct dtype
        let casted = stacked.call_method1("astype", (dtype_str,))?;
        Ok(casted)
    })();

    if let Ok(stacked) = stacked_result {
        // Successfully stacked - process in bulk
        let all_bytes_obj = stacked.call_method0("tobytes")?;
        let all_bytes: Vec<u8> = all_bytes_obj.extract()?;

        // Write type byte + data for each vector
        for i in 0..count {
            result.push(type_byte);
            let start = i * data_bytes_per_vec;
            let end = start + data_bytes_per_vec;
            result.extend_from_slice(&all_bytes[start..end]);
        }
    } else {
        // Fallback: Process individually (for heterogeneous inputs)
        for value in values.iter() {
            // Ensure it's a numpy array
            let arr = if value.hasattr("__array__")? {
                value.clone()
            } else {
                numpy.call_method1("array", (value,))?
            };
            // Convert to target dtype
            let casted = arr.call_method1("astype", (dtype_str,))?;

            // Write type byte
            result.push(type_byte);

            // Get bytes and write data
            let tobytes = casted.call_method0("tobytes")?;
            let bytes: Vec<u8> = tobytes.extract()?;
            result.extend_from_slice(&bytes);
        }
    }

    Ok(PyBytes::new_bound(py, &result).unbind())
}

/// Optimized bulk unpack for fixed-shape vectors
///
/// Uses numpy's frombuffer for efficient batch conversion
pub fn unpack_many_vectors_fixed(
    py: Python,
    data: &[u8],
    count: usize,
    element_type: ElementType,
    shape: &[usize],
) -> Result<Py<PyList>, PyErr> {
    if count == 0 {
        return Ok(PyList::empty_bound(py).unbind());
    }

    // Import numpy
    let numpy = py.import_bound("numpy")?;

    // Calculate sizes
    let elem_count: usize = shape.iter().product();
    let data_bytes_per_vec = elem_count * element_type.byte_size();
    let bytes_per_vec = 1 + data_bytes_per_vec; // type(1) + data

    let expected_size = count * bytes_per_vec;
    if data.len() < expected_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough data: expected {} bytes, got {}",
            expected_size,
            data.len()
        )));
    }

    let list = PyList::empty_bound(py);
    let dtype_str = element_type.to_numpy_dtype();

    // Process each vector
    for i in 0..count {
        let offset = i * bytes_per_vec;

        // Verify type byte
        let type_byte = data[offset];
        if type_byte != element_type as u8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Type mismatch at vector {}: expected {}, got {}",
                i, element_type as u8, type_byte
            )));
        }

        // Extract data bytes
        let data_start = offset + 1;
        let data_end = data_start + data_bytes_per_vec;
        let vec_data = &data[data_start..data_end];

        // Use numpy.frombuffer for efficient conversion
        let array =
            numpy.call_method1("frombuffer", (PyBytes::new_bound(py, vec_data), dtype_str))?;

        // Reshape if multi-dimensional
        let final_array = if shape.len() > 1 {
            let shape_tuple = pyo3::types::PyTuple::new_bound(py, shape);
            array.call_method1("reshape", (shape_tuple,))?
        } else {
            array
        };

        list.append(final_array)?;
    }

    Ok(list.unbind())
}
