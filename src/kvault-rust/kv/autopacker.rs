//! Auto-packing for arbitrary Python objects
//!
//! Automatically detects type and chooses best serialization:
//! 1. Try DataPacker (for numpy arrays and supported types)
//! 2. Fall back to Pickle (for arbitrary Python objects)

use super::header::EncodingType;
use crate::packer::{DataPacker, ElementType, PackerDType};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Auto-packer that detects type and serializes accordingly
pub struct AutoPacker {
    pub try_datapacker: bool,
    pub try_pickle: bool,
}

impl AutoPacker {
    pub fn new(try_pickle: bool) -> Self {
        Self {
            try_datapacker: true,
            try_pickle,
        }
    }

    /// Serialize a Python object automatically
    ///
    /// Returns: (serialized_bytes, encoding_type)
    pub fn serialize(
        &self,
        py: Python,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u8>, EncodingType)> {
        // 1. Check if it's already bytes - keep raw
        if obj.is_instance_of::<PyBytes>() {
            let bytes: Vec<u8> = obj.extract()?;
            return Ok((bytes, EncodingType::Raw));
        }

        // 2. Try DataPacker for numpy arrays
        if self.try_datapacker {
            if let Ok(result) = self.try_serialize_numpy(py, obj) {
                return Ok(result);
            }
        }

        // 3. Try DataPacker for primitives (i64, f64)
        if self.try_datapacker {
            if let Ok(result) = self.try_serialize_primitive(py, obj) {
                return Ok(result);
            }
        }

        // 4. Fall back to pickle for arbitrary objects
        if self.try_pickle {
            let pickle = py.import_bound("pickle")?;
            let pickled = pickle.call_method1("dumps", (obj,))?;
            let bytes: Vec<u8> = pickled.extract()?;
            return Ok((bytes, EncodingType::Pickle));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot auto-pack object: not bytes, numpy array, or picklable",
        ))
    }

    /// Try to serialize as numpy array
    fn try_serialize_numpy(
        &self,
        py: Python,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u8>, EncodingType)> {
        // Check if it has __array__ attribute (numpy array-like)
        if !obj.hasattr("__array__")? {
            return Err(pyo3::exceptions::PyTypeError::new_err("Not a numpy array"));
        }

        // Get shape and dtype
        let shape_obj = obj.getattr("shape")?;
        let shape_tuple = shape_obj.downcast::<pyo3::types::PyTuple>()?;
        let shape: Vec<usize> = shape_tuple
            .iter()
            .map(|x| x.extract::<usize>())
            .collect::<Result<Vec<_>, _>>()?;

        let dtype_obj = obj.getattr("dtype")?;
        let dtype_name_obj = dtype_obj.getattr("name")?;
        let dtype_name: String = dtype_name_obj.extract()?;

        // Map numpy dtype to our ElementType
        let element_type = match dtype_name.as_str() {
            "float32" => ElementType::F32,
            "float64" => ElementType::F64,
            "int32" => ElementType::I32,
            "int64" => ElementType::I64,
            "uint8" => ElementType::U8,
            "uint16" => ElementType::U16,
            "uint32" => ElementType::U32,
            "uint64" => ElementType::U64,
            "int8" => ElementType::I8,
            "int16" => ElementType::I16,
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported numpy dtype: {}",
                    dtype_name
                )))
            }
        };

        // Create a DataPacker with fixed shape (for efficiency)
        let dtype = PackerDType::Vector {
            element_type,
            fixed_shape: Some(shape),
        };
        let packer = DataPacker { dtype };

        // Pack the array
        let packed_bytes = packer.pack_impl(py, obj)?;

        Ok((packed_bytes, EncodingType::DataPacker))
    }

    /// Try to serialize as primitive type (int, float)
    fn try_serialize_primitive(
        &self,
        py: Python,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u8>, EncodingType)> {
        // Try i64
        if let Ok(val) = obj.extract::<i64>() {
            let dtype = PackerDType::I64;
            let packer = DataPacker { dtype };
            let packed_bytes = packer.pack_impl(py, obj)?;
            return Ok((packed_bytes, EncodingType::DataPacker));
        }

        // Try f64
        if let Ok(val) = obj.extract::<f64>() {
            let dtype = PackerDType::F64;
            let packer = DataPacker { dtype };
            let packed_bytes = packer.pack_impl(py, obj)?;
            return Ok((packed_bytes, EncodingType::DataPacker));
        }

        Err(pyo3::exceptions::PyTypeError::new_err("Not a primitive type"))
    }

    /// Deserialize based on encoding type
    pub fn deserialize(
        &self,
        py: Python,
        data: &[u8],
        encoding: EncodingType,
    ) -> PyResult<PyObject> {
        match encoding {
            EncodingType::Raw => {
                // Return as bytes
                Ok(PyBytes::new_bound(py, data).into())
            }
            EncodingType::Pickle => {
                // Unpickle
                let pickle = py.import_bound("pickle")?;
                let unpickled = pickle.call_method1("loads", (PyBytes::new_bound(py, data),))?;
                Ok(unpickled.into())
            }
            EncodingType::DataPacker => {
                // Need to parse the packed data to determine dtype
                // For now, just return bytes (Phase 3 will improve this)
                Ok(PyBytes::new_bound(py, data).into())
            }
            EncodingType::Json => {
                let json = py.import_bound("json")?;
                let text = std::str::from_utf8(data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let decoded = json.call_method1("loads", (text,))?;
                Ok(decoded.into())
            }
            EncodingType::MessagePack => {
                // Use DataPacker's MessagePack implementation
                let dtype = PackerDType::MessagePack { fixed_size: None };
                let packer = DataPacker { dtype };
                packer.unpack_impl(py, data, 0)
            }
            EncodingType::Cbor => {
                // Use DataPacker's CBOR implementation
                let dtype = PackerDType::Cbor {
                    schema: None,
                    fixed_size: None,
                };
                let packer = DataPacker { dtype };
                packer.unpack_impl(py, data, 0)
            }
            EncodingType::Reserved => Err(pyo3::exceptions::PyValueError::new_err(
                "Reserved encoding type",
            )),
        }
    }
}
