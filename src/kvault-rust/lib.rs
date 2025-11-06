// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KohakuVault - SQLite-backed storage with dual interfaces
//!
//! This module exports:
//! - _KVault: Key-value storage with caching
//! - _ColumnVault: Columnar storage with dynamic chunks
//! - DataPacker: Rust-based data serialization
//! - CSB+Tree: Cache-sensitive B+Tree for ordered storage

use pyo3::prelude::*;

mod col;
mod common;
mod kv;
mod packer;
mod tree;

#[pymodule]
fn _kvault(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<kv::_KVault>()?;
    m.add_class::<col::_ColumnVault>()?;
    m.add_class::<packer::DataPacker>()?;

    // Register CSB+Tree (Python object keys & values)
    tree::register_tree_types(m)?;

    Ok(())
}
