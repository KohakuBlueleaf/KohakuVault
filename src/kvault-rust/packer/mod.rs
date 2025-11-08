// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Data packing/unpacking for columnar storage.
//!
//! Supports:
//! - Primitives: i64, f64, string (UTF-8/UTF-16/ASCII/Latin1), bytes
//! - MessagePack: JSON-like schema-less binary format
//! - CBOR: With optional CDDL schema validation
//! - JSON Schema: Validation for MessagePack

mod primitives;
mod structured;
mod vector;
mod vector_bulk;

// Re-export public types
pub use primitives::{pack_bytes, pack_string, unpack_bytes, unpack_string, StringEncoding};
pub use structured::{pack_cbor, pack_messagepack, unpack_cbor, unpack_messagepack};
pub use vector::{pack_vector, unpack_vector, ElementType};

#[cfg(feature = "schema-validation")]
pub use structured::pack_messagepack_validated;

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

    /// MessagePack serialization with optional fixed size
    /// - None: variable-size (grows/shrinks with data)
    /// - Some(n): fixed n bytes (error if exceeds, pad if smaller)
    MessagePack { fixed_size: Option<usize> },

    /// MessagePack with JSON Schema validation
    #[cfg(feature = "schema-validation")]
    MessagePackValidated {
        schema: serde_json::Value,
        compiled_schema: std::sync::Arc<jsonschema::JSONSchema>,
    },

    /// CBOR serialization with optional fixed size
    Cbor {
        schema: Option<String>,
        fixed_size: Option<usize>,
    },

    /// Vector/array with element type and optional fixed shape
    /// - None: arbitrary shape (includes shape in data)
    /// - Some(shape): fixed shape (validates on pack/unpack)
    Vector {
        element_type: ElementType,
        fixed_shape: Option<Vec<usize>>,
    },
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
                        (
                            None,
                            StringEncoding::from_str(parts[1])
                                .map_err(PackerError::InvalidDtype)?,
                        )
                    }
                } else if parts.len() == 3 {
                    // "str:32:utf8" - fixed size with encoding
                    let size = parts[1].parse().map_err(|_| {
                        PackerError::InvalidDtype(format!("Invalid size: {}", parts[1]))
                    })?;
                    let encoding =
                        StringEncoding::from_str(parts[2]).map_err(PackerError::InvalidDtype)?;
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

            "msgpack" | "messagepack" => {
                // Parse: "msgpack" or "msgpack:128"
                let fixed_size = if parts.len() > 1 {
                    Some(parts[1].parse().map_err(|_| {
                        PackerError::InvalidDtype(format!("Invalid size: {}", parts[1]))
                    })?)
                } else {
                    None
                };
                Ok(Self::MessagePack { fixed_size })
            }

            "cbor" => {
                // Parse: "cbor" or "cbor:128"
                let fixed_size = if parts.len() > 1 {
                    Some(parts[1].parse().map_err(|_| {
                        PackerError::InvalidDtype(format!("Invalid size: {}", parts[1]))
                    })?)
                } else {
                    None
                };
                Ok(Self::Cbor { schema: None, fixed_size })
            }

            "vec" | "vector" => {
                // Parse: "vec:f32", "vec:f32:128", "vec:i64:10:20", etc.
                if parts.len() < 2 {
                    return Err(PackerError::InvalidDtype(
                        "Vector type requires element type (e.g., vec:f32)".to_string(),
                    ));
                }

                let element_type = ElementType::from_str(parts[1]).ok_or_else(|| {
                    PackerError::InvalidDtype(format!(
                        "Unknown element type '{}'. Expected f32, f64, i32, i64, u8, etc.",
                        parts[1]
                    ))
                })?;

                let fixed_shape = if parts.len() > 2 {
                    // Parse fixed shape dimensions
                    let mut shape = Vec::new();
                    for dim_str in &parts[2..] {
                        let dim = dim_str.parse::<usize>().map_err(|_| {
                            PackerError::InvalidDtype(format!("Invalid dimension: {}", dim_str))
                        })?;
                        if dim == 0 {
                            return Err(PackerError::InvalidDtype(
                                "Dimension must be greater than 0".to_string(),
                            ));
                        }
                        shape.push(dim);
                    }
                    Some(shape)
                } else {
                    None
                };

                Ok(Self::Vector { element_type, fixed_shape })
            }

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
            Self::MessagePack { fixed_size: Some(size) } => *size,
            Self::MessagePack { fixed_size: None } => 0,
            #[cfg(feature = "schema-validation")]
            Self::MessagePackValidated { .. } => 0,
            Self::Cbor { fixed_size: Some(size), .. } => *size,
            Self::Cbor { fixed_size: None, .. } => 0,
            Self::Vector { element_type, fixed_shape } => {
                if let Some(shape) = fixed_shape {
                    vector::calculate_vector_size(*element_type, shape)
                } else {
                    0 // Arbitrary shape is variable-size
                }
            }
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
    pub(crate) dtype: PackerDType,
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
        Ok(Self {
            dtype: PackerDType::Cbor { schema: schema.map(|s| s.to_string()), fixed_size: None },
        })
    }

    /// Pack single value to bytes
    pub fn pack(&self, py: Python, value: &Bound<PyAny>) -> PyResult<Py<PyBytes>> {
        let bytes = self.pack_impl(py, value)?;
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }

    /// Pack multiple values to concatenated bytes
    pub fn pack_many(&self, py: Python, values: &Bound<PyList>) -> PyResult<Py<PyBytes>> {
        // Use optimized bulk path for fixed-shape vectors
        if let PackerDType::Vector { element_type, fixed_shape: Some(ref shape) } = &self.dtype {
            return vector_bulk::pack_many_vectors_fixed(py, values, *element_type, shape);
        }

        // Generic path for all other types
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
        // Use optimized bulk path for fixed-shape vectors
        if let PackerDType::Vector { element_type, fixed_shape: Some(ref shape) } = &self.dtype {
            if let Some(n) = count {
                return vector_bulk::unpack_many_vectors_fixed(py, data, n, *element_type, shape);
            }
        }

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
    pub(crate) fn pack_impl(&self, _py: Python, value: &Bound<PyAny>) -> Result<Vec<u8>, PyErr> {
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
                pack_string(&s, *encoding, *fixed_size)
            }

            PackerDType::Bytes { fixed_size } => {
                let bytes: Vec<u8> = value.extract()?;
                pack_bytes(&bytes, *fixed_size)
            }

            PackerDType::MessagePack { fixed_size } => {
                let bytes = pack_messagepack(value)?;
                if let Some(size) = fixed_size {
                    pack_bytes(&bytes, Some(*size))
                } else {
                    Ok(bytes)
                }
            }

            #[cfg(feature = "schema-validation")]
            PackerDType::MessagePackValidated { schema: _schema, compiled_schema } => {
                pack_messagepack_validated(value, compiled_schema)
            }

            PackerDType::Cbor { schema, fixed_size } => {
                let bytes = pack_cbor(value, schema.as_deref())?;
                if let Some(size) = fixed_size {
                    pack_bytes(&bytes, Some(*size))
                } else {
                    Ok(bytes)
                }
            }

            PackerDType::Vector { element_type, fixed_shape } => {
                pack_vector(value, *element_type, fixed_shape.as_deref())
            }
        }
    }

    pub(crate) fn unpack_impl(
        &self,
        py: Python,
        data: &[u8],
        offset: usize,
    ) -> Result<PyObject, PyErr> {
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
                unpack_string(py, data, offset, *encoding, *fixed_size)
            }

            PackerDType::Bytes { fixed_size } => unpack_bytes(py, data, offset, *fixed_size),

            PackerDType::MessagePack { fixed_size } => {
                if let Some(size) = fixed_size {
                    // Fixed-size: read exact bytes
                    if offset + size > data.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Not enough data",
                        ));
                    }
                    // MessagePack is self-delimiting, so just try to decode
                    // The deserializer will stop at the end of the msgpack data
                    unpack_messagepack(py, &data[offset..offset + size])
                } else {
                    // Variable-size: read from offset to end
                    unpack_messagepack(py, &data[offset..])
                }
            }

            #[cfg(feature = "schema-validation")]
            PackerDType::MessagePackValidated { .. } => unpack_messagepack(py, &data[offset..]),

            PackerDType::Cbor { fixed_size, .. } => {
                if let Some(size) = fixed_size {
                    // Fixed-size: read exact bytes
                    if offset + size > data.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Not enough data",
                        ));
                    }
                    // CBOR is self-delimiting, so just try to decode
                    // The deserializer will stop at the end of the cbor data
                    unpack_cbor(py, &data[offset..offset + size])
                } else {
                    // Variable-size: read from offset to end
                    unpack_cbor(py, &data[offset..])
                }
            }

            PackerDType::Vector { element_type, fixed_shape } => {
                unpack_vector(py, data, offset, fixed_shape.as_deref(), *element_type)
            }
        }
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

        // Test msgpack (variable-size)
        let dtype = PackerDType::from_str("msgpack").unwrap();
        assert!(dtype.is_varsize());
        assert_eq!(dtype.elem_size(), 0);

        // Test msgpack (fixed-size)
        let dtype = PackerDType::from_str("msgpack:256").unwrap();
        assert!(!dtype.is_varsize());
        assert_eq!(dtype.elem_size(), 256);

        // Test cbor (fixed-size)
        let dtype = PackerDType::from_str("cbor:128").unwrap();
        assert_eq!(dtype.elem_size(), 128);
    }
}
