// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Data packing/unpacking for columnar storage.
//!
//! Supports:
//! - Primitives: i64, f64, string (UTF-8/UTF-16/ASCII/Latin1), bytes
//! - MessagePack: JSON-like schema-less binary format
//! - CBOR: With optional CDDL schema validation
//! - JSON Schema: Validation for MessagePack

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PackerError {
    #[error("Serialization error: {0}")]
    Serialize(String),

    #[error("Deserialization error: {0}")]
    Deserialize(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Invalid dtype: {0}")]
    InvalidDtype(String),

    #[error("Encoding error: {0}")]
    Encoding(String),
}

impl From<PackerError> for PyErr {
    fn from(err: PackerError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

/// String encoding options
#[derive(Debug, Clone, Copy)]
pub enum StringEncoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    Ascii,
    Latin1,
}

impl StringEncoding {
    fn from_str(s: &str) -> Result<Self, PackerError> {
        match s.to_lowercase().as_str() {
            "utf8" | "utf-8" => Ok(Self::Utf8),
            "utf16le" | "utf-16le" => Ok(Self::Utf16Le),
            "utf16be" | "utf-16be" => Ok(Self::Utf16Be),
            "ascii" => Ok(Self::Ascii),
            "latin1" | "iso-8859-1" => Ok(Self::Latin1),
            _ => Err(PackerError::InvalidDtype(format!("Unknown encoding: {}", s))),
        }
    }
}

/// Data packer type variants
#[derive(Debug, Clone)]
pub enum PackerDType {
    /// Fixed 64-bit signed integer
    I64,

    /// Fixed 64-bit float
    F64,

    /// String with encoding and optional fixed size
    String {
        encoding: StringEncoding,
        fixed_size: Option<usize>, // If None, variable-size
    },

    /// Raw bytes with optional fixed size
    Bytes { fixed_size: Option<usize> },

    /// MessagePack serialization (schema-less)
    MessagePack,

    /// MessagePack with JSON Schema validation
    #[cfg(feature = "schema-validation")]
    MessagePackValidated {
        schema: serde_json::Value,
        compiled_schema: std::sync::Arc<jsonschema::JSONSchema>,
    },

    /// CBOR serialization (optionally with CDDL schema)
    Cbor { schema: Option<String> },
}

impl PackerDType {
    /// Parse dtype string: "i64", "f64", "str", "str:utf8", "str:32:utf8", "bytes", "bytes:128", "msgpack", "cbor"
    pub fn from_str(dtype_str: &str) -> Result<Self, PackerError> {
        let parts: Vec<&str> = dtype_str.split(':').collect();

        match parts[0] {
            "i64" => Ok(Self::I64),
            "f64" => Ok(Self::F64),

            "str" | "string" => {
                // Parse: "str", "str:utf8", "str:32", "str:32:utf8"
                let (fixed_size, encoding) = if parts.len() == 1 {
                    // "str" - variable UTF-8
                    (None, StringEncoding::Utf8)
                } else if parts.len() == 2 {
                    // Could be "str:utf8" or "str:32"
                    if let Ok(size) = parts[1].parse::<usize>() {
                        // "str:32" - fixed size, UTF-8
                        (Some(size), StringEncoding::Utf8)
                    } else {
                        // "str:utf8" - variable size, specified encoding
                        (None, StringEncoding::from_str(parts[1])?)
                    }
                } else if parts.len() == 3 {
                    // "str:32:utf8" - fixed size with encoding
                    let size = parts[1].parse().map_err(|_| {
                        PackerError::InvalidDtype(format!("Invalid size: {}", parts[1]))
                    })?;
                    let encoding = StringEncoding::from_str(parts[2])?;
                    (Some(size), encoding)
                } else {
                    return Err(PackerError::InvalidDtype(format!(
                        "Invalid str dtype format: {}",
                        dtype_str
                    )));
                };

                Ok(Self::String { encoding, fixed_size })
            }

            "bytes" => {
                let fixed_size = if parts.len() > 1 {
                    Some(parts[1].parse().map_err(|_| {
                        PackerError::InvalidDtype(format!("Invalid size: {}", parts[1]))
                    })?)
                } else {
                    None
                };
                Ok(Self::Bytes { fixed_size })
            }

            "msgpack" | "messagepack" => Ok(Self::MessagePack),

            "cbor" => Ok(Self::Cbor { schema: None }),

            _ => Err(PackerError::InvalidDtype(format!("Unknown dtype: {}", dtype_str))),
        }
    }

    /// Get element size (0 for variable-size)
    pub fn elem_size(&self) -> usize {
        match self {
            Self::I64 => 8,
            Self::F64 => 8,
            Self::String { fixed_size: Some(size), .. } => *size,
            Self::String { fixed_size: None, .. } => 0,
            Self::Bytes { fixed_size: Some(size) } => *size,
            Self::Bytes { fixed_size: None } => 0,
            Self::MessagePack => 0,
            #[cfg(feature = "schema-validation")]
            Self::MessagePackValidated { .. } => 0,
            Self::Cbor { .. } => 0,
        }
    }

    /// Check if variable-size
    pub fn is_varsize(&self) -> bool {
        self.elem_size() == 0
    }
}

/// Main data packer class exposed to Python
#[pyclass]
pub struct DataPacker {
    dtype: PackerDType,
}

#[pymethods]
impl DataPacker {
    /// Create new packer from dtype string
    ///
    /// # Examples (Python)
    /// ```python
    /// packer = DataPacker("i64")
    /// packer = DataPacker("f64")
    /// packer = DataPacker("str:utf8")
    /// packer = DataPacker("str:32:utf8")  # Fixed 32 bytes UTF-8
    /// packer = DataPacker("bytes:128")    # Fixed 128 bytes
    /// packer = DataPacker("msgpack")      # MessagePack
    /// packer = DataPacker("cbor")         # CBOR
    /// ```
    #[new]
    fn new(dtype_str: &str) -> PyResult<Self> {
        let dtype = PackerDType::from_str(dtype_str)?;
        Ok(Self { dtype })
    }

    /// Create MessagePack packer with JSON Schema validation
    #[cfg(feature = "schema-validation")]
    #[staticmethod]
    fn with_json_schema(schema: &Bound<'_, PyDict>) -> PyResult<Self> {
        let schema_value: serde_json::Value = pythonize::depythonize(schema.as_any())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let compiled = jsonschema::JSONSchema::compile(&schema_value).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Schema compilation failed: {}",
                e
            ))
        })?;

        Ok(Self {
            dtype: PackerDType::MessagePackValidated {
                schema: schema_value,
                compiled_schema: std::sync::Arc::new(compiled),
            },
        })
    }

    /// Create CBOR packer with optional CDDL schema
    #[staticmethod]
    #[pyo3(signature = (schema=None))]
    fn with_cddl_schema(schema: Option<&str>) -> PyResult<Self> {
        Ok(Self { dtype: PackerDType::Cbor { schema: schema.map(|s| s.to_string()) } })
    }

    /// Pack single value to bytes
    pub fn pack(&self, py: Python, value: &Bound<PyAny>) -> PyResult<Py<PyBytes>> {
        let bytes = self.pack_impl(py, value)?;
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }

    /// Pack multiple values to concatenated bytes
    pub fn pack_many(&self, py: Python, values: &Bound<PyList>) -> PyResult<Py<PyBytes>> {
        let mut result = Vec::new();

        for value in values.iter() {
            let bytes = self.pack_impl(py, &value)?;
            result.extend_from_slice(&bytes);
        }

        Ok(PyBytes::new_bound(py, &result).unbind())
    }

    /// Unpack single value from bytes at offset
    pub fn unpack(&self, py: Python, data: &[u8], offset: usize) -> PyResult<PyObject> {
        self.unpack_impl(py, data, offset)
    }

    /// Unpack multiple values from bytes
    ///
    /// For fixed-size types: Uses count to determine number of values
    /// For variable-size types: Uses offsets list to determine boundaries
    #[pyo3(signature = (data, count=None, offsets=None))]
    pub fn unpack_many(
        &self,
        py: Python,
        data: &[u8],
        count: Option<usize>,
        offsets: Option<Vec<usize>>,
    ) -> PyResult<Py<PyList>> {
        let list = PyList::empty_bound(py);
        let elem_size = self.dtype.elem_size();

        if elem_size == 0 {
            // Variable-size type: need offsets
            if let Some(offset_list) = offsets {
                // offsets contains the start position of each element
                // For N elements, we need N offsets (start of each element)
                // The end of last element is len(data)
                for i in 0..offset_list.len() {
                    let start = offset_list[i];
                    let end = if i + 1 < offset_list.len() {
                        offset_list[i + 1]
                    } else {
                        data.len()
                    };

                    if end > data.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Offset out of bounds: {} > {}",
                            end,
                            data.len()
                        )));
                    }

                    // Extract the slice for this element
                    let element_data = &data[start..end];
                    // Unpack from the beginning of the slice (offset=0)
                    let value = self.unpack_impl(py, element_data, 0)?;
                    list.append(value)?;
                }
                Ok(list.unbind())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Variable-size types require offsets parameter",
                ))
            }
        } else {
            // Fixed-size type: use count
            let count = count.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Fixed-size types require count parameter",
                )
            })?;

            for i in 0..count {
                let offset = i * elem_size;
                let value = self.unpack_impl(py, data, offset)?;
                list.append(value)?;
            }

            Ok(list.unbind())
        }
    }

    /// Get element size (0 for variable-size)
    #[getter]
    pub fn elem_size(&self) -> usize {
        self.dtype.elem_size()
    }

    /// Check if variable-size
    #[getter]
    fn is_varsize(&self) -> bool {
        self.dtype.is_varsize()
    }

    /// Get dtype string representation
    fn __repr__(&self) -> String {
        format!("DataPacker({:?})", self.dtype)
    }
}

// Implementation methods (not exposed to Python)
impl DataPacker {
    fn pack_impl(&self, _py: Python, value: &Bound<PyAny>) -> Result<Vec<u8>, PyErr> {
        match &self.dtype {
            PackerDType::I64 => {
                let val: i64 = value.extract()?;
                Ok(val.to_le_bytes().to_vec())
            }

            PackerDType::F64 => {
                let val: f64 = value.extract()?;
                Ok(val.to_le_bytes().to_vec())
            }

            PackerDType::String { encoding, fixed_size } => {
                let s: String = value.extract()?;
                self.pack_string(&s, *encoding, *fixed_size)
            }

            PackerDType::Bytes { fixed_size } => {
                let bytes: Vec<u8> = value.extract()?;
                self.pack_bytes(&bytes, *fixed_size)
            }

            PackerDType::MessagePack => self.pack_messagepack(value),

            #[cfg(feature = "schema-validation")]
            PackerDType::MessagePackValidated { schema: _schema, compiled_schema } => {
                self.pack_messagepack_validated(value, compiled_schema)
            }

            PackerDType::Cbor { schema } => self.pack_cbor(value, schema.as_deref()),
        }
    }

    fn unpack_impl(&self, py: Python, data: &[u8], offset: usize) -> Result<PyObject, PyErr> {
        match &self.dtype {
            PackerDType::I64 => {
                if offset + 8 > data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not enough data"));
                }
                let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap();
                let val = i64::from_le_bytes(bytes);
                Ok(val.into_py(py))
            }

            PackerDType::F64 => {
                if offset + 8 > data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not enough data"));
                }
                let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap();
                let val = f64::from_le_bytes(bytes);
                Ok(val.into_py(py))
            }

            PackerDType::String { encoding, fixed_size } => {
                self.unpack_string(py, data, offset, *encoding, *fixed_size)
            }

            PackerDType::Bytes { fixed_size } => self.unpack_bytes(py, data, offset, *fixed_size),

            PackerDType::MessagePack => self.unpack_messagepack(py, &data[offset..]),

            #[cfg(feature = "schema-validation")]
            PackerDType::MessagePackValidated { .. } => {
                self.unpack_messagepack(py, &data[offset..])
            }

            PackerDType::Cbor { .. } => self.unpack_cbor(py, &data[offset..]),
        }
    }

    // Helper methods for specific types

    fn pack_string(
        &self,
        s: &str,
        encoding: StringEncoding,
        fixed_size: Option<usize>,
    ) -> Result<Vec<u8>, PyErr> {
        let encoded = match encoding {
            StringEncoding::Utf8 => s.as_bytes().to_vec(),
            StringEncoding::Utf16Le => s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect(),
            StringEncoding::Utf16Be => s.encode_utf16().flat_map(|c| c.to_be_bytes()).collect(),
            StringEncoding::Ascii => {
                if s.is_ascii() {
                    s.as_bytes().to_vec()
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "String contains non-ASCII characters",
                    ));
                }
            }
            StringEncoding::Latin1 => {
                // Latin1 maps code points 0-255 directly to bytes
                if s.chars().all(|c| (c as u32) <= 255) {
                    s.chars().map(|c| c as u8).collect()
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "String contains characters outside Latin1 range",
                    ));
                }
            }
        };

        if let Some(size) = fixed_size {
            self.pack_bytes(&encoded, Some(size))
        } else {
            Ok(encoded)
        }
    }

    fn pack_bytes(&self, bytes: &[u8], fixed_size: Option<usize>) -> Result<Vec<u8>, PyErr> {
        if let Some(size) = fixed_size {
            if bytes.len() > size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Data too long: {} > {}",
                    bytes.len(),
                    size
                )));
            }

            // Pad with zeros
            let mut result = bytes.to_vec();
            result.resize(size, 0);
            Ok(result)
        } else {
            Ok(bytes.to_vec())
        }
    }

    fn unpack_string(
        &self,
        py: Python,
        data: &[u8],
        offset: usize,
        encoding: StringEncoding,
        fixed_size: Option<usize>,
    ) -> Result<PyObject, PyErr> {
        let size = fixed_size.unwrap_or(data.len() - offset);
        if offset + size > data.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not enough data"));
        }

        let bytes = &data[offset..offset + size];

        let string = match encoding {
            StringEncoding::Utf8 => String::from_utf8(bytes.to_vec()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "UTF-8 decode error: {}",
                    e
                ))
            })?,
            StringEncoding::Utf16Le => {
                let u16_vec: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                String::from_utf16(&u16_vec).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "UTF-16 decode error: {}",
                        e
                    ))
                })?
            }
            StringEncoding::Utf16Be => {
                let u16_vec: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect();
                String::from_utf16(&u16_vec).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "UTF-16 decode error: {}",
                        e
                    ))
                })?
            }
            StringEncoding::Ascii => {
                if bytes.iter().all(|&b| b <= 127) {
                    String::from_utf8(bytes.to_vec()).unwrap()
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Data contains non-ASCII bytes",
                    ));
                }
            }
            StringEncoding::Latin1 => bytes.iter().map(|&b| b as char).collect(),
        };

        // Trim null padding for fixed-size strings
        let trimmed = if fixed_size.is_some() {
            string.trim_end_matches('\0').to_string()
        } else {
            string
        };

        Ok(trimmed.into_py(py))
    }

    fn unpack_bytes(
        &self,
        py: Python,
        data: &[u8],
        offset: usize,
        fixed_size: Option<usize>,
    ) -> Result<PyObject, PyErr> {
        let size = fixed_size.unwrap_or(data.len() - offset);
        if offset + size > data.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not enough data"));
        }

        let bytes = &data[offset..offset + size];
        Ok(PyBytes::new_bound(py, bytes).into())
    }

    fn pack_messagepack(&self, value: &Bound<PyAny>) -> Result<Vec<u8>, PyErr> {
        // Convert Python object to serde_json::Value
        let json_value: serde_json::Value =
            pythonize::depythonize(value.as_any()).map_err(|e| {
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

    #[cfg(feature = "schema-validation")]
    fn pack_messagepack_validated(
        &self,
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

    fn unpack_messagepack(&self, py: Python, data: &[u8]) -> Result<PyObject, PyErr> {
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

    fn pack_cbor(&self, value: &Bound<PyAny>, _schema: Option<&str>) -> Result<Vec<u8>, PyErr> {
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

    fn unpack_cbor(&self, py: Python, data: &[u8]) -> Result<PyObject, PyErr> {
        let json_value: serde_json::Value = ciborium::de::from_reader(data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CBOR decoding failed: {}", e))
        })?;

        pythonize::pythonize(py, &json_value)
            .map(|bound| bound.unbind())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parsing() {
        // Test i64
        let dtype = PackerDType::from_str("i64").unwrap();
        assert_eq!(dtype.elem_size(), 8);
        assert!(!dtype.is_varsize());

        // Test f64
        let dtype = PackerDType::from_str("f64").unwrap();
        assert_eq!(dtype.elem_size(), 8);

        // Test variable string
        let dtype = PackerDType::from_str("str:utf8").unwrap();
        assert_eq!(dtype.elem_size(), 0);
        assert!(dtype.is_varsize());

        // Test fixed string
        let dtype = PackerDType::from_str("str:32:utf8").unwrap();
        assert_eq!(dtype.elem_size(), 32);
        assert!(!dtype.is_varsize());

        // Test bytes
        let dtype = PackerDType::from_str("bytes:128").unwrap();
        assert_eq!(dtype.elem_size(), 128);

        // Test msgpack
        let dtype = PackerDType::from_str("msgpack").unwrap();
        assert!(dtype.is_varsize());
    }

    #[test]
    fn test_string_encoding_from_str() {
        assert!(matches!(StringEncoding::from_str("utf8").unwrap(), StringEncoding::Utf8));
        assert!(matches!(StringEncoding::from_str("UTF-8").unwrap(), StringEncoding::Utf8));
        assert!(matches!(StringEncoding::from_str("ascii").unwrap(), StringEncoding::Ascii));
        assert!(StringEncoding::from_str("invalid").is_err());
    }
}
