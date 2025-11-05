// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Structured data formats: MessagePack and CBOR
//!
//! Provides serialization/deserialization for:
//! - MessagePack (with optional JSON Schema validation)
//! - CBOR (with optional CDDL schema)

use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Pack Python object to MessagePack format.
pub fn pack_messagepack(value: &Bound<PyAny>) -> Result<Vec<u8>, PyErr> {
    // Convert Python object to serde_json::Value
    let json_value: serde_json::Value = pythonize::depythonize(value.as_any()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to convert Python object: {}",
            e
        ))
    })?;

    // Serialize to MessagePack
    rmp_serde::to_vec(&json_value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "MessagePack encoding failed: {}",
            e
        ))
    })
}

/// Pack Python object to MessagePack format with JSON Schema validation.
#[cfg(feature = "schema-validation")]
pub fn pack_messagepack_validated(
    value: &Bound<PyAny>,
    schema: &jsonschema::JSONSchema,
) -> Result<Vec<u8>, PyErr> {
    let json_value: serde_json::Value = pythonize::depythonize(value.as_any())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Validate against schema
    if let Err(errors) = schema.validate(&json_value) {
        let error_msgs: Vec<String> = errors.map(|e| e.to_string()).collect();
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Validation errors: {}",
            error_msgs.join(", ")
        )));
    }

    // Serialize to MessagePack
    rmp_serde::to_vec(&json_value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "MessagePack encoding failed: {}",
            e
        ))
    })
}

/// Unpack MessagePack data to Python object.
pub fn unpack_messagepack(py: Python, data: &[u8]) -> Result<PyObject, PyErr> {
    // Deserialize from MessagePack
    let json_value: serde_json::Value = rmp_serde::from_slice(data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "MessagePack decoding failed: {}",
            e
        ))
    })?;

    // Convert to Python object
    pythonize::pythonize(py, &json_value)
        .map(|bound| bound.unbind())
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to convert to Python: {}",
                e
            ))
        })
}

/// Pack Python object to CBOR format.
pub fn pack_cbor(value: &Bound<PyAny>, _schema: Option<&str>) -> Result<Vec<u8>, PyErr> {
    let json_value: serde_json::Value = pythonize::depythonize(value.as_any())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Serialize to CBOR
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&json_value, &mut buf).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CBOR encoding failed: {}", e))
    })?;

    // TODO: CDDL validation if schema provided

    Ok(buf)
}

/// Unpack CBOR data to Python object.
pub fn unpack_cbor(py: Python, data: &[u8]) -> Result<PyObject, PyErr> {
    let json_value: serde_json::Value = ciborium::de::from_reader(data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CBOR decoding failed: {}", e))
    })?;

    pythonize::pythonize(py, &json_value)
        .map(|bound| bound.unbind())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
