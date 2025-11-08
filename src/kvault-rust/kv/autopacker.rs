//! Auto-packing for arbitrary Python objects
//!
//! Automatically detects type and chooses best serialization:
//!
//! Priority order:
//! 1. bytes → Raw (no encoding/header)
//! 2. Wrapped types (MsgPack, Json, Cbor, Pickle) → Specified encoding
//! 3. **DataPacker-supported types**:
//!    - numpy array → vec:* (f32, i64, etc.)
//!    - int → i64
//!    - float → f64
//! 4. str → UTF-8 encode (simple, automatic decode)
//! 5. dict/list → Try MessagePack, fallback to Pickle if fails
//! 6. Custom objects → Pickle (last resort)

use super::header::EncodingType;
use crate::packer::{DataPacker, ElementType, PackerDType};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// Auto-packer that detects type and serializes accordingly
pub struct AutoPacker {
    pub use_pickle_fallback: bool,
}

impl AutoPacker {
    pub fn new(use_pickle_fallback: bool) -> Self {
        Self { use_pickle_fallback }
    }

    /// Serialize a Python object automatically
    ///
    /// Returns: (serialized_bytes, encoding_type)
    pub fn serialize(
        &self,
        py: Python,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u8>, EncodingType)> {
        // 1. Check if it's bytes - keep raw (no header!)
        if obj.is_instance_of::<PyBytes>() {
            let bytes: Vec<u8> = obj.extract()?;
            return Ok((bytes, EncodingType::Raw));
        }

        // 2. Check if it's a wrapped type (MsgPack, Json, Cbor, Pickle)
        if obj.hasattr("encoding_name")? && obj.hasattr("data")? {
            let encoding_name: String = obj.getattr("encoding_name")?.extract()?;
            let data = obj.getattr("data")?;

            return match encoding_name.as_str() {
                "msgpack" => {
                    let packer =
                        DataPacker { dtype: PackerDType::MessagePack { fixed_size: None } };
                    let bytes = packer.pack_impl(py, &data)?;
                    Ok((bytes, EncodingType::MessagePack))
                }
                "json" => {
                    let json_mod = py.import_bound("json")?;
                    let json_str = json_mod.call_method1("dumps", (&data,))?;
                    let bytes: Vec<u8> = json_str.extract::<String>()?.into_bytes();
                    Ok((bytes, EncodingType::Json))
                }
                "cbor" => {
                    let packer =
                        DataPacker { dtype: PackerDType::Cbor { schema: None, fixed_size: None } };
                    let bytes = packer.pack_impl(py, &data)?;
                    Ok((bytes, EncodingType::Cbor))
                }
                "pickle" => {
                    let pickle = py.import_bound("pickle")?;
                    let pickled = pickle.call_method1("dumps", (&data,))?;
                    let bytes: Vec<u8> = pickled.extract()?;
                    Ok((bytes, EncodingType::Pickle))
                }
                _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown encoding wrapper: {}",
                    encoding_name
                ))),
            };
        }

        // 3. Try numpy array → DataPacker vec:*
        if obj.hasattr("__array__")? {
            if let Ok(result) = self.serialize_numpy(py, obj) {
                return Ok(result);
            }
        }

        // 4. Try int → DataPacker i64
        if let Ok(_val) = obj.extract::<i64>() {
            let packer = DataPacker { dtype: PackerDType::I64 };
            let bytes = packer.pack_impl(py, obj)?;
            return Ok((bytes, EncodingType::DataPacker));
        }

        // 5. Try float → DataPacker f64
        if let Ok(_val) = obj.extract::<f64>() {
            let packer = DataPacker { dtype: PackerDType::F64 };
            let bytes = packer.pack_impl(py, obj)?;
            return Ok((bytes, EncodingType::DataPacker));
        }

        // 6. Try str → UTF-8 encode (simple, always decodable)
        if let Ok(s) = obj.extract::<String>() {
            let bytes = s.into_bytes();
            return Ok((bytes, EncodingType::Utf8String));
        }

        // 7. Try dict/list → MessagePack first, fallback to Pickle if fails
        if obj.is_instance_of::<PyDict>() || obj.is_instance_of::<PyList>() {
            // Try MessagePack first (more efficient)
            let packer = DataPacker { dtype: PackerDType::MessagePack { fixed_size: None } };
            if let Ok(bytes) = packer.pack_impl(py, obj) {
                return Ok((bytes, EncodingType::MessagePack));
            }

            // MessagePack failed (unsupported type in dict/list) - fallback to Pickle
            if self.use_pickle_fallback {
                let pickle = py.import_bound("pickle")?;
                let pickled = pickle.call_method1("dumps", (obj,))?;
                let bytes: Vec<u8> = pickled.extract()?;
                return Ok((bytes, EncodingType::Pickle));
            }

            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Cannot serialize dict/list: MessagePack failed and pickle disabled",
            ));
        }

        // 8. Last resort: Pickle (for custom objects)
        if self.use_pickle_fallback {
            let pickle = py.import_bound("pickle")?;
            let pickled = pickle.call_method1("dumps", (obj,))?;
            let bytes: Vec<u8> = pickled.extract()?;
            return Ok((bytes, EncodingType::Pickle));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot auto-pack object: unsupported type and pickle fallback disabled",
        ))
    }

    /// Serialize numpy array to DataPacker vec:* format
    fn serialize_numpy(
        &self,
        py: Python,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u8>, EncodingType)> {
        // Get shape (not used since we use arbitrary shape format, but needed to validate)
        let shape_obj = obj.getattr("shape")?;
        let shape_tuple = shape_obj.downcast::<pyo3::types::PyTuple>()?;
        let _shape: Vec<usize> = shape_tuple
            .iter()
            .map(|x| x.extract::<usize>())
            .collect::<Result<Vec<_>, _>>()?;

        // Get dtype
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

        // Use arbitrary shape format (includes shape in data)
        // This allows deserializer to reconstruct shape without metadata
        let dtype = PackerDType::Vector {
            element_type,
            fixed_shape: None, // Arbitrary shape
        };
        let packer = DataPacker { dtype };

        // Pack the array
        let bytes = packer.pack_impl(py, obj)?;

        Ok((bytes, EncodingType::DataPacker))
    }

    /// Deserialize based on encoding type
    ///
    /// Returns actual Python object (not bytes!)
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
            EncodingType::DataPacker => {
                // Determine what type of DataPacker data this is based on length and content
                if data.is_empty() {
                    return Ok(PyBytes::new_bound(py, data).into());
                }

                // Primitives are exactly 8 bytes (i64, f64)
                // Try both and use heuristic to choose
                if data.len() == 8 {
                    let bytes_array: [u8; 8] = data.try_into().unwrap();

                    // Try as f64
                    let as_f64 = f64::from_le_bytes(bytes_array);
                    // Try as i64
                    let as_i64 = i64::from_le_bytes(bytes_array);

                    // Heuristic: if f64 interpretation is finite and has decimal part, prefer f64
                    // Otherwise prefer i64
                    if as_f64.is_finite() && as_f64.fract().abs() > 1e-9 {
                        return Ok(as_f64.into_py(py));
                    } else if as_i64.abs() < 1_000_000_000_000 {
                        // Reasonable int range
                        return Ok(as_i64.into_py(py));
                    } else {
                        // Large number - could be float stored as int bits
                        return Ok(as_f64.into_py(py));
                    }
                }

                // Vectors have type byte (0x01-0x0A) and are always longer than 8 bytes
                // Minimum vector size: type(1) + data(>8 bytes) OR type(1) + ndim(1) + shape(4+) + data
                if data.len() > 8 && ElementType::from_u8(data[0]).is_some() {
                    let elem_type = ElementType::from_u8(data[0]).unwrap();
                    if let Ok(obj) = crate::packer::unpack_vector(py, data, 0, None, elem_type) {
                        return Ok(obj);
                    }
                }

                // Try string (no specific marker, just valid UTF-8)
                if let Ok(s) = std::str::from_utf8(data) {
                    return Ok(s.into_py(py));
                }

                // Fallback to raw bytes
                Ok(PyBytes::new_bound(py, data).into())
            }
            EncodingType::Utf8String => {
                // Decode UTF-8 bytes to string
                let s = std::str::from_utf8(data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(s.into_py(py))
            }
            EncodingType::MessagePack => {
                let packer = DataPacker { dtype: PackerDType::MessagePack { fixed_size: None } };
                packer.unpack_impl(py, data, 0)
            }
            EncodingType::Json => {
                let json_mod = py.import_bound("json")?;
                let text = std::str::from_utf8(data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let decoded = json_mod.call_method1("loads", (text,))?;
                Ok(decoded.into())
            }
            EncodingType::Cbor => {
                let packer =
                    DataPacker { dtype: PackerDType::Cbor { schema: None, fixed_size: None } };
                packer.unpack_impl(py, data, 0)
            }
            EncodingType::Pickle => {
                let pickle = py.import_bound("pickle")?;
                let unpickled = pickle.call_method1("loads", (PyBytes::new_bound(py, data),))?;
                Ok(unpickled.into())
            }
            EncodingType::Reserved => {
                Err(pyo3::exceptions::PyValueError::new_err("Reserved encoding type"))
            }
        }
    }
}
