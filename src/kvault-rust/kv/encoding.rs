//! Encoding helpers for KVault values with headers
//!
//! Provides methods to encode/decode values with optional headers.

use super::_KVault;
use super::header::{EncodingType, Header, HEADER_SIZE};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::Ordering;

impl _KVault {
    /// Encode value with header if headers are enabled and encoding is not Raw
    ///
    /// Smart behavior:
    /// - If headers disabled: return raw bytes
    /// - If headers enabled AND encoding is Raw: return raw bytes (for media file compat)
    /// - If headers enabled AND encoding is not Raw: prepend header
    pub(crate) fn encode_value(&self, data: &[u8], encoding: EncodingType) -> Vec<u8> {
        let use_headers = self.use_headers.load(Ordering::Relaxed);

        if !use_headers || encoding == EncodingType::Raw {
            // Keep raw bytes as-is (no header)
            // This ensures media files can be previewed by external tools
            data.to_vec()
        } else {
            // Add header for encoded data
            let header = Header::new(encoding);
            let mut result = Vec::with_capacity(HEADER_SIZE + data.len());
            result.extend_from_slice(&header.encode());
            result.extend_from_slice(data);
            result
        }
    }

    /// Decode value, stripping header if present
    ///
    /// Returns: (data, Option<Header>)
    /// - If no header: returns (original_bytes, None)
    /// - If header: returns (data_without_header, Some(header))
    pub(crate) fn decode_value(&self, bytes: &[u8]) -> Result<(Vec<u8>, Option<Header>), String> {
        match Header::decode(bytes)? {
            Some(header) => {
                if bytes.len() < HEADER_SIZE {
                    return Err("Value too short for header".to_string());
                }
                Ok((bytes[HEADER_SIZE..].to_vec(), Some(header)))
            }
            None => {
                // No header, return original bytes
                Ok((bytes.to_vec(), None))
            }
        }
    }

    /// Get value and return with header information
    ///
    /// Returns: (data, Option<Header>)
    #[allow(dead_code)] // Used in Phase 3 (auto-unpacking)
    pub(crate) fn get_with_header_info(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyBytes>, Option<Header>)> {
        let raw_bytes = self.get_impl(py, key)?;
        let bytes_vec: Vec<u8> = raw_bytes.extract(py)?;

        let (data, header) = self
            .decode_value(&bytes_vec)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok((PyBytes::new_bound(py, &data).unbind(), header))
    }

    /// Put value with specific encoding type
    ///
    /// This will add a header if headers are enabled (unless encoding is Raw)
    #[allow(dead_code)] // Used in Phase 3 (auto-packing)
    pub(crate) fn put_with_encoding(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &[u8],
        encoding: EncodingType,
    ) -> PyResult<()> {
        let k = super::ops::to_key_bytes(py, key)?;
        let encoded_value = self.encode_value(value, encoding);

        // Use direct write (bypass cache for now to keep it simple)
        self.write_direct(&k, &encoded_value)?;
        Ok(())
    }
}
