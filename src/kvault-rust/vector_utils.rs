//! Utility functions for vector operations with sqlite-vec
//!
//! This module provides helpers for converting between Rust Vec<T> and
//! SQLite BLOB format compatible with sqlite-vec extension.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zerocopy::AsBytes;

/// Vector element types supported by sqlite-vec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorType {
    Float32,
    Int8,
    Bit,
}

impl VectorType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" | "float32" => Some(VectorType::Float32),
            "i8" | "int8" => Some(VectorType::Int8),
            "bit" => Some(VectorType::Bit),
            _ => None,
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            VectorType::Float32 => "float32",
            VectorType::Int8 => "int8",
            VectorType::Bit => "bit",
        }
    }
}

/// Convert Vec<f32> to BLOB format for sqlite-vec
pub fn vec_f32_to_blob(vec: &[f32]) -> Vec<u8> {
    vec.as_bytes().to_vec()
}

/// Convert BLOB to Vec<f32>
pub fn blob_to_vec_f32(blob: &[u8]) -> Result<Vec<f32>, String> {
    if !blob.len().is_multiple_of(4) {
        return Err(format!(
            "Invalid blob length {} for float32 vector (must be divisible by 4)",
            blob.len()
        ));
    }

    let float_count = blob.len() / 4;
    let mut result = Vec::with_capacity(float_count);

    for i in 0..float_count {
        let bytes = [
            blob[i * 4],
            blob[i * 4 + 1],
            blob[i * 4 + 2],
            blob[i * 4 + 3],
        ];
        result.push(f32::from_le_bytes(bytes));
    }

    Ok(result)
}

/// Convert Python list/array to Vec<f32>
pub fn py_to_vec_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try to extract as sequence
    if let Ok(seq) = obj.downcast::<pyo3::types::PyList>() {
        let mut result = Vec::with_capacity(seq.len());
        for item in seq.iter() {
            result.push(item.extract::<f32>()?);
        }
        return Ok(result);
    }

    // Try as tuple
    if let Ok(tuple) = obj.downcast::<pyo3::types::PyTuple>() {
        let mut result = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            result.push(item.extract::<f32>()?);
        }
        return Ok(result);
    }

    // Try as bytes (direct BLOB)
    if let Ok(bytes) = obj.downcast::<PyBytes>() {
        let blob = bytes.as_bytes();
        return blob_to_vec_f32(blob).map_err(pyo3::exceptions::PyValueError::new_err);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected list, tuple, or bytes of floats",
    ))
}

/// Normalize vector to unit length (L2 normalization)
pub fn normalize_l2(vec: &mut [f32]) {
    let magnitude: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in vec.iter_mut() {
            *x /= magnitude;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_f32_conversion() {
        let vec = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = vec_f32_to_blob(&vec);
        assert_eq!(blob.len(), 16); // 4 floats * 4 bytes

        let recovered = blob_to_vec_f32(&blob).unwrap();
        assert_eq!(recovered, vec);
    }

    #[test]
    fn test_normalize_l2() {
        let mut vec = vec![3.0f32, 4.0];
        normalize_l2(&mut vec);

        // 3-4-5 triangle: magnitude = 5
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);

        // Verify unit length
        let magnitude: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }
}
