//! KVault value header system for encoding type detection
//!
//! Header format (10 bytes total):
//! ```
//! |magic_k(2)|version(1)|encoding(1)|flags(1)|reserved(3)|magic_q(2)|data...|
//! ```
//!
//! - magic_k: Opening magic bytes (0x89, 0x4B) - "K" for KohakuVault
//!   - In future: 0x4B can encode header length if reserved bytes are used
//! - version: Format version (0x01 for v1)
//! - encoding: Encoding type (see EncodingType enum)
//! - flags: Bit flags (compression, encryption, etc.)
//! - reserved: Reserved for future use (must be 0x00 for v1)
//! - magic_q: Closing magic bytes (0x56, 0x4B) - "VK" for validation
//!   - Validates header is real, not accidental match in user data
//!
//! Total: 10 bytes overhead

/// Header size in bytes
pub const HEADER_SIZE: usize = 10;

/// Opening magic bytes
pub const MAGIC_K: [u8; 2] = [0x89, 0x4B]; // 0x89 is high-bit set (unlikely ASCII), K for KohakuVault

/// Closing magic bytes (validation)
pub const MAGIC_Q: [u8; 2] = [0x56, 0x4B]; // VK for validation

/// Current header version
pub const HEADER_VERSION: u8 = 0x01;

/// Encoding types for value data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingType {
    /// Raw bytes (no encoding, backward compatible)
    Raw = 0x00,

    /// Python pickle serialization
    Pickle = 0x01,

    /// DataPacker serialization (dtype follows in extended header)
    DataPacker = 0x02,

    /// JSON encoding
    Json = 0x03,

    /// MessagePack encoding
    MessagePack = 0x04,

    /// CBOR encoding
    Cbor = 0x05,

    /// UTF-8 string (simple encode/decode)
    Utf8String = 0x06,

    /// Reserved for future use
    Reserved = 0xFF,
}

impl EncodingType {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(EncodingType::Raw),
            0x01 => Some(EncodingType::Pickle),
            0x02 => Some(EncodingType::DataPacker),
            0x03 => Some(EncodingType::Json),
            0x04 => Some(EncodingType::MessagePack),
            0x05 => Some(EncodingType::Cbor),
            0x06 => Some(EncodingType::Utf8String),
            0xFF => Some(EncodingType::Reserved),
            _ => None,
        }
    }

    #[allow(dead_code)] // Used for logging/debugging
    pub fn to_str(self) -> &'static str {
        match self {
            EncodingType::Raw => "raw",
            EncodingType::Pickle => "pickle",
            EncodingType::DataPacker => "datapacker",
            EncodingType::Json => "json",
            EncodingType::MessagePack => "messagepack",
            EncodingType::Cbor => "cbor",
            EncodingType::Utf8String => "utf8string",
            EncodingType::Reserved => "reserved",
        }
    }
}

/// Header flags (bit flags)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderFlags(u8);

impl HeaderFlags {
    pub const NONE: u8 = 0x00;
    #[allow(dead_code)] // Used in Phase 3 (compression)
    pub const COMPRESSED: u8 = 0x01; // bit 0: zlib compression
    #[allow(dead_code)] // Future use
    pub const ENCRYPTED: u8 = 0x02; // bit 1: encryption (future)

    pub fn new(flags: u8) -> Self {
        Self(flags)
    }

    #[allow(dead_code)] // Used in Phase 3
    pub fn is_compressed(self) -> bool {
        (self.0 & Self::COMPRESSED) != 0
    }

    #[allow(dead_code)] // Future use
    pub fn is_encrypted(self) -> bool {
        (self.0 & Self::ENCRYPTED) != 0
    }

    pub fn as_u8(self) -> u8 {
        self.0
    }
}

/// Parsed header information
#[derive(Debug, Clone)]
pub struct Header {
    pub version: u8,
    pub encoding: EncodingType,
    pub flags: HeaderFlags,
    pub reserved: [u8; 3],
}

impl Header {
    /// Create a new header with standard settings
    pub fn new(encoding: EncodingType) -> Self {
        Self {
            version: HEADER_VERSION,
            encoding,
            flags: HeaderFlags::new(HeaderFlags::NONE),
            reserved: [0x00, 0x00, 0x00],
        }
    }

    /// Create header with flags
    pub fn with_flags(encoding: EncodingType, flags: u8) -> Self {
        Self {
            version: HEADER_VERSION,
            encoding,
            flags: HeaderFlags::new(flags),
            reserved: [0x00, 0x00, 0x00],
        }
    }

    /// Encode header to 10 bytes
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];

        bytes[0..2].copy_from_slice(&MAGIC_K);
        bytes[2] = self.version;
        bytes[3] = self.encoding as u8;
        bytes[4] = self.flags.as_u8();
        bytes[5..8].copy_from_slice(&self.reserved);
        bytes[8..10].copy_from_slice(&MAGIC_Q);

        bytes
    }

    /// Decode header from bytes
    ///
    /// Returns None if not a valid header (for backward compatibility)
    pub fn decode(bytes: &[u8]) -> Result<Option<Self>, String> {
        // Need at least 10 bytes
        if bytes.len() < HEADER_SIZE {
            return Ok(None);
        }

        // Check opening magic
        if bytes[0..2] != MAGIC_K {
            return Ok(None); // Not a header, assume old format
        }

        // Check closing magic for validation
        if bytes[8..10] != MAGIC_Q {
            return Err("Invalid header: closing magic mismatch".to_string());
        }

        let version = bytes[2];
        if version != HEADER_VERSION {
            return Err(format!("Unsupported header version: {}", version));
        }

        let encoding = EncodingType::from_u8(bytes[3])
            .ok_or_else(|| format!("Unknown encoding type: {}", bytes[3]))?;

        let flags = HeaderFlags::new(bytes[4]);

        let reserved = [bytes[5], bytes[6], bytes[7]];

        // For v1, reserved bytes must be 0x00
        if reserved != [0x00, 0x00, 0x00] {
            return Err("Invalid header: reserved bytes must be 0x00 for v1".to_string());
        }

        Ok(Some(Self { version, encoding, flags, reserved }))
    }

    /// Check if bytes have a valid header
    #[allow(dead_code)] // Used in tests and Phase 3
    pub fn has_header(bytes: &[u8]) -> bool {
        bytes.len() >= HEADER_SIZE && bytes[0..2] == MAGIC_K && bytes[8..10] == MAGIC_Q
    }
}

/// Build header bytes for a given encoding
#[allow(dead_code)] // Used in Phase 3
pub fn build_header(encoding: EncodingType) -> Vec<u8> {
    Header::new(encoding).encode().to_vec()
}

/// Build header bytes with flags
#[allow(dead_code)] // Used in Phase 3
pub fn build_header_with_flags(encoding: EncodingType, flags: u8) -> Vec<u8> {
    Header::with_flags(encoding, flags).encode().to_vec()
}

/// Extract data from value with header
///
/// Returns (data, header) tuple
#[allow(dead_code)] // Used in Phase 3
pub fn extract_data(bytes: &[u8]) -> Result<(&[u8], Option<Header>), String> {
    if let Some(header) = Header::decode(bytes)? {
        if bytes.len() < HEADER_SIZE {
            return Err("Value too short for header".to_string());
        }
        Ok((&bytes[HEADER_SIZE..], Some(header)))
    } else {
        // No header, return all bytes as data
        Ok((bytes, None))
    }
}

/// Prepend header to data
#[allow(dead_code)] // Used in Phase 3
pub fn prepend_header(encoding: EncodingType, data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(HEADER_SIZE + data.len());
    result.extend_from_slice(&Header::new(encoding).encode());
    result.extend_from_slice(data);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_encode_decode() {
        let header = Header::new(EncodingType::Pickle);
        let encoded = header.encode();

        assert_eq!(encoded.len(), HEADER_SIZE);
        assert_eq!(&encoded[0..2], &MAGIC_K);
        assert_eq!(&encoded[8..10], &MAGIC_Q);
        assert_eq!(encoded[2], HEADER_VERSION);
        assert_eq!(encoded[3], EncodingType::Pickle as u8);

        let decoded = Header::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.version, HEADER_VERSION);
        assert_eq!(decoded.encoding, EncodingType::Pickle);
    }

    #[test]
    fn test_header_magic_validation() {
        // Valid header
        let mut bytes = [0u8; 10];
        bytes[0..2].copy_from_slice(&MAGIC_K);
        bytes[2] = HEADER_VERSION;
        bytes[3] = EncodingType::Raw as u8;
        bytes[8..10].copy_from_slice(&MAGIC_Q);

        assert!(Header::has_header(&bytes));
        assert!(Header::decode(&bytes).unwrap().is_some());

        // Missing closing magic
        bytes[8..10].copy_from_slice(&[0xFF, 0xFF]);
        assert!(!Header::has_header(&bytes));
        assert!(Header::decode(&bytes).is_err());
    }

    #[test]
    fn test_header_backward_compatibility() {
        // Old format (no header)
        let old_data = b"raw bytes without header";

        assert!(!Header::has_header(old_data));
        let result = Header::decode(old_data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_data() {
        // New format with header
        let header = Header::new(EncodingType::Pickle);
        let data = b"pickled data here";
        let mut value = header.encode().to_vec();
        value.extend_from_slice(data);

        let (extracted_data, extracted_header) = extract_data(&value).unwrap();
        assert_eq!(extracted_data, data);
        assert!(extracted_header.is_some());
        assert_eq!(extracted_header.unwrap().encoding, EncodingType::Pickle);

        // Old format (no header)
        let old_value = b"raw bytes";
        let (extracted_data, extracted_header) = extract_data(old_value).unwrap();
        assert_eq!(extracted_data, old_value);
        assert!(extracted_header.is_none());
    }

    #[test]
    fn test_prepend_header() {
        let data = b"test data";
        let with_header = prepend_header(EncodingType::DataPacker, data);

        assert_eq!(with_header.len(), HEADER_SIZE + data.len());
        assert_eq!(&with_header[0..2], &MAGIC_K);
        assert_eq!(&with_header[8..10], &MAGIC_Q);
        assert_eq!(&with_header[HEADER_SIZE..], data);
    }

    #[test]
    fn test_encoding_types() {
        for encoding in [
            EncodingType::Raw,
            EncodingType::Pickle,
            EncodingType::DataPacker,
            EncodingType::Json,
            EncodingType::MessagePack,
            EncodingType::Cbor,
        ] {
            let byte = encoding as u8;
            let decoded = EncodingType::from_u8(byte).unwrap();
            assert_eq!(decoded, encoding);
        }
    }

    #[test]
    fn test_header_flags() {
        let flags = HeaderFlags::new(HeaderFlags::COMPRESSED | HeaderFlags::ENCRYPTED);

        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());

        let flags_none = HeaderFlags::new(HeaderFlags::NONE);
        assert!(!flags_none.is_compressed());
        assert!(!flags_none.is_encrypted());
    }
}
