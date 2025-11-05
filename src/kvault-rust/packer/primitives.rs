// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! Primitive type packing/unpacking: i64, f64, bytes, strings
//!
//! Handles encoding and decoding of basic data types with various
//! string encodings (UTF-8, UTF-16, ASCII, Latin1).

use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "utf8" | "utf-8" => Ok(Self::Utf8),
            "utf16le" | "utf-16le" => Ok(Self::Utf16Le),
            "utf16be" | "utf-16be" => Ok(Self::Utf16Be),
            "ascii" => Ok(Self::Ascii),
            "latin1" | "iso-8859-1" => Ok(Self::Latin1),
            _ => Err(format!("Unknown encoding: {}", s)),
        }
    }
}

/// Pack a string with specified encoding and optional fixed size.
pub fn pack_string(
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
        pack_bytes(&encoded, Some(size))
    } else {
        Ok(encoded)
    }
}

/// Pack bytes with optional fixed size (zero-padding if needed).
pub fn pack_bytes(bytes: &[u8], fixed_size: Option<usize>) -> Result<Vec<u8>, PyErr> {
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

/// Unpack a string from bytes with specified encoding and optional fixed size.
pub fn unpack_string(
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
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("UTF-8 decode error: {}", e))
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

/// Unpack bytes from data with optional fixed size.
pub fn unpack_bytes(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_encoding_from_str() {
        assert!(matches!(StringEncoding::from_str("utf8").unwrap(), StringEncoding::Utf8));
        assert!(matches!(StringEncoding::from_str("UTF-8").unwrap(), StringEncoding::Utf8));
        assert!(matches!(StringEncoding::from_str("ascii").unwrap(), StringEncoding::Ascii));
        assert!(StringEncoding::from_str("invalid").is_err());
    }
}
