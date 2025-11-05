//! Variable-size column operations
//!
//! This module contains all variable-size column specific implementations.
//! Variable-size columns (bytes, str, msgpack, cbor) use adaptive chunking with an index.
//!
//! Index Format: Each element has a 12-byte index entry (chunk_id: i32, start: i32, end: i32)
//! that points to its location in the data chunks.

// Variable-size column operations will be implemented here
// Functions to move from col/mod.rs:
// - read_adaptive, batch_read_varsize, batch_read_varsize_unpacked
// - update_varsize_element, update_varsize_slice
// - append_raw_adaptive, extend_adaptive, append_raw_adaptive_cached, extend_adaptive_cached
// - delete_adaptive
// - Internal helpers: update_index_entry_internal, shift_chunk_indices_after, etc.
