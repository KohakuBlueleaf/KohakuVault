//! Vector/array packing and unpacking for DataPacker
//!
//! Supports arbitrary and fixed-shape arrays with efficient binary format:
//! - Arbitrary: |type(1)|ndim(1)|shape(ndim*4)|data...|
//! - Fixed:     |type(1)|data...|

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Element types for vectors/arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    F32 = 0x01,
    F64 = 0x02,
    I32 = 0x03,
    I64 = 0x04,
    U8 = 0x05,
    U16 = 0x06,
    U32 = 0x07,
    U64 = 0x08,
    I8 = 0x09,
    I16 = 0x0A,
}

impl ElementType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" | "float32" => Some(ElementType::F32),
            "f64" | "float64" => Some(ElementType::F64),
            "i32" | "int32" => Some(ElementType::I32),
            "i64" | "int64" => Some(ElementType::I64),
            "u8" | "uint8" => Some(ElementType::U8),
            "u16" | "uint16" => Some(ElementType::U16),
            "u32" | "uint32" => Some(ElementType::U32),
            "u64" | "uint64" => Some(ElementType::U64),
            "i8" | "int8" => Some(ElementType::I8),
            "i16" | "int16" => Some(ElementType::I16),
            _ => None,
        }
    }

    pub fn to_numpy_dtype(self) -> &'static str {
        match self {
            ElementType::F32 => "float32",
            ElementType::F64 => "float64",
            ElementType::I32 => "int32",
            ElementType::I64 => "int64",
            ElementType::U8 => "uint8",
            ElementType::U16 => "uint16",
            ElementType::U32 => "uint32",
            ElementType::U64 => "uint64",
            ElementType::I8 => "int8",
            ElementType::I16 => "int16",
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            ElementType::F32 => 4,
            ElementType::F64 => 8,
            ElementType::I32 => 4,
            ElementType::I64 => 8,
            ElementType::U8 => 1,
            ElementType::U16 => 2,
            ElementType::U32 => 4,
            ElementType::U64 => 8,
            ElementType::I8 => 1,
            ElementType::I16 => 2,
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(ElementType::F32),
            0x02 => Some(ElementType::F64),
            0x03 => Some(ElementType::I32),
            0x04 => Some(ElementType::I64),
            0x05 => Some(ElementType::U8),
            0x06 => Some(ElementType::U16),
            0x07 => Some(ElementType::U32),
            0x08 => Some(ElementType::U64),
            0x09 => Some(ElementType::I8),
            0x0A => Some(ElementType::I16),
            _ => None,
        }
    }
}

/// Pack numpy array to binary format
///
/// Format:
/// - Arbitrary shape: |type(1)|ndim(1)|shape(ndim*4)|data...|
/// - Fixed shape:     |type(1)|data...|
pub fn pack_vector(
    value: &Bound<PyAny>,
    element_type: ElementType,
    fixed_shape: Option<&[usize]>,
) -> Result<Vec<u8>, PyErr> {
    // Extract numpy array or list
    let (shape, flat_data) = extract_array_data(value, element_type)?;

    // Validate shape if fixed
    if let Some(expected_shape) = fixed_shape {
        if shape != expected_shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: expected {:?}, got {:?}",
                expected_shape, shape
            )));
        }
    }

    // Calculate total size
    let elem_count: usize = shape.iter().product();
    let data_bytes = elem_count * element_type.byte_size();

    let mut result = if fixed_shape.is_some() {
        // Fixed shape: |type(1)|data...|
        Vec::with_capacity(1 + data_bytes)
    } else {
        // Arbitrary shape: |type(1)|ndim(1)|shape(ndim*4)|data...|
        let header_size = 2 + shape.len() * 4;
        Vec::with_capacity(header_size + data_bytes)
    };

    // Write type byte
    result.push(element_type as u8);

    // Write shape if arbitrary
    if fixed_shape.is_none() {
        result.push(shape.len() as u8); // ndim
        for &dim in &shape {
            result.extend_from_slice(&(dim as u32).to_le_bytes());
        }
    }

    // Write data
    result.extend_from_slice(&flat_data);

    Ok(result)
}

/// Unpack binary data to numpy array
pub fn unpack_vector(
    py: Python,
    data: &[u8],
    offset: usize,
    fixed_shape: Option<&[usize]>,
    _expected_element_type: ElementType,
) -> Result<PyObject, PyErr> {
    if data.len() <= offset {
        return Err(pyo3::exceptions::PyValueError::new_err("Not enough data"));
    }

    let mut pos = offset;

    // Read type
    let element_type = ElementType::from_u8(data[pos]).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown element type: {}", data[pos]))
    })?;
    pos += 1;

    // Read shape
    let shape = if let Some(s) = fixed_shape {
        s.to_vec()
    } else {
        // Arbitrary shape: read ndim and shape
        if pos >= data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Not enough data for ndim"));
        }
        let ndim = data[pos] as usize;
        pos += 1;

        let mut shape_vec = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            if pos + 4 > data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("Not enough data for shape"));
            }
            let dim = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            shape_vec.push(dim);
            pos += 4;
        }
        shape_vec
    };

    // Read data
    let elem_count: usize = shape.iter().product();
    let data_bytes = elem_count * element_type.byte_size();

    if pos + data_bytes > data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough data: need {} bytes, have {}",
            data_bytes,
            data.len() - pos
        )));
    }

    // Import numpy
    let numpy = py.import_bound("numpy")?;

    // Convert data to Python list based on element type
    let flat_list = create_python_list(py, &data[pos..pos + data_bytes], element_type, elem_count)?;

    // Get numpy dtype string
    let dtype_str = element_type.to_numpy_dtype();

    // Create numpy array from list with explicit dtype
    let array = numpy.call_method1("array", (flat_list,))?;
    let array = array.call_method1("astype", (dtype_str,))?;

    // Reshape if needed
    if shape.len() > 1 {
        let shape_tuple = pyo3::types::PyTuple::new_bound(py, &shape);
        let reshaped = array.call_method1("reshape", (shape_tuple,))?;
        Ok(reshaped.into())
    } else {
        Ok(array.into())
    }
}

/// Extract array data from numpy array or Python list
fn extract_array_data(
    value: &Bound<PyAny>,
    element_type: ElementType,
) -> Result<(Vec<usize>, Vec<u8>), PyErr> {
    // Try to get numpy array
    let numpy = value.py().import_bound("numpy")?;

    // Convert to numpy array if not already
    let array = if value.hasattr("__array__")? {
        value.clone()
    } else {
        numpy.call_method1("array", (value,))?
    };

    // Get shape
    let shape_obj = array.getattr("shape")?;
    let shape_tuple = shape_obj.downcast::<pyo3::types::PyTuple>()?;
    let shape: Vec<usize> = shape_tuple
        .iter()
        .map(|x| x.extract::<usize>().unwrap())
        .collect();

    // Convert to correct dtype and get bytes
    let dtype_str = element_type.to_numpy_dtype();
    let casted = array.call_method1("astype", (dtype_str,))?;
    let tobytes = casted.call_method0("tobytes")?;
    let bytes: Vec<u8> = tobytes.extract()?;

    Ok((shape, bytes))
}

/// Create Python list from raw bytes
fn create_python_list(
    py: Python,
    data: &[u8],
    element_type: ElementType,
    count: usize,
) -> Result<Py<PyList>, PyErr> {
    let list = PyList::empty_bound(py);

    match element_type {
        ElementType::F32 => {
            for i in 0..count {
                let offset = i * 4;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let val = f32::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::F64 => {
            for i in 0..count {
                let offset = i * 8;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                let val = f64::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::I32 => {
            for i in 0..count {
                let offset = i * 4;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let val = i32::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::I64 => {
            for i in 0..count {
                let offset = i * 8;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                let val = i64::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::U8 => {
            for &byte in data.iter().take(count) {
                list.append(byte)?;
            }
        }
        ElementType::U16 => {
            for i in 0..count {
                let offset = i * 2;
                let bytes = [data[offset], data[offset + 1]];
                let val = u16::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::U32 => {
            for i in 0..count {
                let offset = i * 4;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let val = u32::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::U64 => {
            for i in 0..count {
                let offset = i * 8;
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                let val = u64::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
        ElementType::I8 => {
            for &byte in data.iter().take(count) {
                list.append(byte as i8)?;
            }
        }
        ElementType::I16 => {
            for i in 0..count {
                let offset = i * 2;
                let bytes = [data[offset], data[offset + 1]];
                let val = i16::from_le_bytes(bytes);
                list.append(val)?;
            }
        }
    }

    Ok(list.unbind())
}

/// Calculate total size needed for fixed-shape vector
pub fn calculate_vector_size(element_type: ElementType, shape: &[usize]) -> usize {
    let elem_count: usize = shape.iter().product();
    1 + elem_count * element_type.byte_size() // type(1) + data
}
