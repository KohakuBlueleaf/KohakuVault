// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KohakuVault - SQLite-backed storage with dual interfaces
//!
//! This module exports:
//! - _KVault: Key-value storage with caching
//! - _ColumnVault: Columnar storage with dynamic chunks
//! - DataPacker: Rust-based data serialization
//! - CSB+Tree: Cache-sensitive B+Tree for ordered storage
//! - _VectorKVault: Vector similarity search storage

use pyo3::prelude::*;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;
use std::ffi::c_char;

mod col;
mod common;
mod kv;
mod packer;
mod skiplist;
mod textvault;
mod tree;
mod vector_utils;
mod vkvault;

// Initialize sqlite-vec extension globally
// This must be called before any SQLite connections are opened
#[allow(non_upper_case_globals)]
static mut vec_initialized: bool = false;

fn init_sqlite_vec() {
    unsafe {
        if !vec_initialized {
            sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *const c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(sqlite3_vec_init as *const ())));
            vec_initialized = true;
        }
    }
}

#[pymodule]
fn _kvault(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize sqlite-vec extension before anything else
    init_sqlite_vec();

    m.add_class::<kv::_KVault>()?;
    m.add_class::<col::_ColumnVault>()?;
    m.add_class::<packer::DataPacker>()?;

    // Register CSB+Tree (Python object keys & values)
    tree::register_tree_types(m)?;

    // Register SkipList
    skiplist::register_skiplist_types(m)?;

    // Register VectorKVault
    vkvault::register_vkvault_types(m)?;

    // Register TextVault
    textvault::register_textvault_types(m)?;

    Ok(())
}
