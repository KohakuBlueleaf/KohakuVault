//! Core VectorKVault struct and initialization

use crate::kv::autopacker::AutoPacker;
use crate::kv::header::EncodingType;
use crate::vector_utils::{normalize_l2, py_to_vec_f32, vec_f32_to_blob, VectorType};
use crate::vkvault::metrics::SimilarityMetric;
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::Connection;
use std::sync::atomic::{AtomicBool, Ordering};

/// VectorKVault - Vector similarity search with arbitrary values
///
/// Uses sqlite-vec's vec0 virtual table for vector storage and search
pub struct VectorKVault {
    pub(crate) conn: Mutex<Connection>,
    pub(crate) table: String,
    pub(crate) dimensions: usize,
    pub(crate) metric: SimilarityMetric,
    pub(crate) vector_type: VectorType,
    pub(crate) use_headers: AtomicBool, // Enable header format for new writes
    pub(crate) auto_packer: Mutex<Option<AutoPacker>>, // Auto-packing for arbitrary objects
}

impl VectorKVault {
    /// Create or open a VectorKVault
    pub fn new(
        path: &str,
        table: &str,
        dimensions: usize,
        metric: &str,
        vector_type: &str,
    ) -> PyResult<Self> {
        let metric = SimilarityMetric::from_str(metric).map_err(PyValueError::new_err)?;

        let vec_type = VectorType::from_str(vector_type).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown vector type: {}", vector_type))
        })?;

        // Validate metric compatibility
        if !metric.is_compatible_with(vec_type) {
            return Err(PyValueError::new_err(format!(
                "Metric '{}' is not compatible with vector type '{}'",
                metric.to_str(),
                vec_type.to_str()
            )));
        }

        // Validate bit vector dimensions
        if vec_type == VectorType::Bit && !dimensions.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Bit vector dimensions must be divisible by 8, got {}",
                dimensions
            )));
        }

        let conn = Connection::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {}", e)))?;

        // Configure SQLite for performance
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set WAL mode: {}", e)))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set synchronous: {}", e)))?;
        conn.pragma_update(None, "temp_store", "MEMORY")
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set temp_store: {}", e)))?;

        let vkv = Self {
            conn: Mutex::new(conn),
            table: table.to_string(),
            dimensions,
            metric,
            vector_type: vec_type,
            use_headers: AtomicBool::new(true), // DEFAULT: headers enabled for auto-packing
            auto_packer: Mutex::new(Some(AutoPacker::new(true))), // DEFAULT: auto-pack enabled
        };

        vkv.create_tables()?;
        Ok(vkv)
    }

    /// Create tables if they don't exist
    fn create_tables(&self) -> PyResult<()> {
        let conn = self.conn.lock();

        // Determine vector type string for SQL
        let vector_sql_type = match self.vector_type {
            VectorType::Float32 => format!("float[{}]", self.dimensions),
            VectorType::Int8 => format!("int8[{}]", self.dimensions),
            VectorType::Bit => format!("bit[{}]", self.dimensions),
        };

        // Create vec0 virtual table with value_ref metadata column
        let create_vec_table = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(
                value_ref INTEGER,
                vector {} distance_metric={}
            )",
            &self.table,
            vector_sql_type,
            self.metric.to_str()
        );

        conn.execute(&create_vec_table, [])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create vec0 table: {}", e)))?;

        // Create values blob table
        let create_values_table = format!(
            "CREATE TABLE IF NOT EXISTS {}_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value BLOB NOT NULL
            )",
            &self.table
        );

        conn.execute(&create_values_table, []).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create values table: {}", e))
        })?;

        Ok(())
    }

    /// Prepare vector blob with validation
    pub(crate) fn prepare_vector_blob(&self, vector: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        match self.vector_type {
            VectorType::Float32 => {
                let mut vec = py_to_vec_f32(vector)?;

                if vec.len() != self.dimensions {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} dimensions, got {}",
                        self.dimensions,
                        vec.len()
                    )));
                }

                // Auto-normalize for cosine similarity
                if self.metric == SimilarityMetric::Cosine {
                    normalize_l2(&mut vec);
                }

                Ok(vec_f32_to_blob(&vec))
            }
            VectorType::Int8 => Err(PyValueError::new_err(
                "Int8 vectors not yet supported. Use float32 or implement quantization first."
                    .to_string(),
            )),
            VectorType::Bit => Err(PyValueError::new_err(
                "Bit vectors not yet supported. Use float32 or implement quantization first."
                    .to_string(),
            )),
        }
    }

    /// Extract bytes from Python object
    pub(crate) fn extract_bytes(&self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        if let Ok(bytes) = obj.downcast::<PyBytes>() {
            return Ok(bytes.as_bytes().to_vec());
        }

        if let Ok(bytearray) = obj.extract::<Vec<u8>>() {
            return Ok(bytearray);
        }

        Err(pyo3::exceptions::PyTypeError::new_err("Expected bytes or bytearray"))
    }

    /// Get table information
    pub fn info(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("table", &self.table)?;
        dict.set_item("dimensions", self.dimensions)?;
        dict.set_item("metric", self.metric.to_str())?;
        dict.set_item("vector_type", self.vector_type.to_str())?;
        dict.set_item("count", self.count()?)?;

        Ok(dict.unbind())
    }

    /// Get total count of vectors
    pub fn count(&self) -> PyResult<i64> {
        let conn = self.conn.lock();
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {}", &self.table), [], |row| row.get(0))
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        Ok(count)
    }

    /// Check if ID exists
    pub fn exists(&self, id: i64) -> PyResult<bool> {
        let conn = self.conn.lock();
        let count: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {} WHERE rowid = ?", &self.table),
                rusqlite::params![id],
                |row| row.get(0),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {}", e)))?;

        Ok(count > 0)
    }

    /// Encode value with header if headers are enabled and encoding is not Raw
    ///
    /// Exactly same behavior as KVault:
    /// - If headers disabled: return raw bytes
    /// - If headers enabled AND encoding is Raw: return raw bytes
    /// - If headers enabled AND encoding is not Raw: prepend header
    pub(crate) fn encode_value(&self, data: &[u8], encoding: EncodingType) -> Vec<u8> {
        let use_headers = self.use_headers.load(Ordering::Relaxed);

        if !use_headers || encoding == EncodingType::Raw {
            // Keep raw bytes as-is (no header)
            data.to_vec()
        } else {
            // Add header for encoded data
            use crate::kv::header::{Header, HEADER_SIZE};
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
    pub(crate) fn decode_value(
        &self,
        bytes: &[u8],
    ) -> Result<(Vec<u8>, Option<crate::kv::header::Header>), String> {
        use crate::kv::header::{Header, HEADER_SIZE};

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

    /// Enable auto-packing (allows arbitrary Python objects as values)
    pub fn enable_auto_pack(&self, use_pickle: bool) -> PyResult<()> {
        let mut guard = self.auto_packer.lock();
        *guard = Some(AutoPacker::new(use_pickle));

        // Also enable headers (required for auto-pack)
        self.use_headers.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Disable auto-packing (return to bytes-only mode)
    pub fn disable_auto_pack(&self) {
        let mut guard = self.auto_packer.lock();
        *guard = None;
    }

    /// Check if auto-packing is enabled
    pub fn auto_pack_enabled(&self) -> bool {
        self.auto_packer.lock().is_some()
    }

    /// Enable header format for new writes
    pub fn enable_headers(&self) {
        self.use_headers.store(true, Ordering::Relaxed);
    }

    /// Disable header format (return to raw bytes mode)
    pub fn disable_headers(&self) {
        self.use_headers.store(false, Ordering::Relaxed);
    }

    /// Check if headers are enabled
    pub fn headers_enabled(&self) -> bool {
        self.use_headers.load(Ordering::Relaxed)
    }

    /// Decode and deserialize value (EXACTLY SAME AS KVault)
    ///
    /// This method auto-decodes based on header if auto-pack is enabled
    pub(crate) fn decode_and_deserialize(&self, py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
        // Check if auto-pack is enabled for auto-decoding
        let has_auto_pack = self.auto_packer.lock().is_some();

        if has_auto_pack {
            // Auto-decode based on header
            let (decoded_data, header) = self
                .decode_value(data)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;

            if let Some(h) = header {
                // Has header - auto-decode based on encoding type
                let auto_pack_guard = self.auto_packer.lock();
                if let Some(ref packer) = *auto_pack_guard {
                    return packer.deserialize(py, &decoded_data, h.encoding);
                }
            }

            // No header - return raw bytes
            Ok(PyBytes::new_bound(py, data).into())
        } else {
            // Auto-pack disabled - always return raw bytes
            Ok(PyBytes::new_bound(py, data).into())
        }
    }
}
