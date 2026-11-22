// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! KohakuVault - SQLite-backed KV, column, text, and vector storage.
//!
//! Usable both as a pure-Rust library crate (`default-features = false`) and as
//! the Python extension module `kohakuvault._kvault` (`--features python`).

pub use rusqlite;

pub mod col;
pub mod common;
pub mod kv;
#[cfg(feature = "python")]
mod packer;
pub mod skiplist;
pub mod textvault;
pub mod tree;
#[cfg(feature = "python")]
mod vector_utils;
#[cfg(feature = "python")]
mod vkvault;

pub use col::{ColError, ColResult, ColumnInfo, ColumnVault};
pub use common::VaultError;
pub use kv::{EncodingType, Header, HeaderFlags, KVault, KVaultOptions, HEADER_SIZE};
pub use textvault::{escape_fts5_query, TextRecord, TextSearchHit, TextVault};

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use rusqlite::ffi::sqlite3_auto_extension;
    use sqlite_vec::sqlite3_vec_init;
    use std::ffi::c_char;

    use crate::{col, kv, packer, skiplist, textvault, tree, vkvault};

    // Register sqlite-vec once before concurrent connection creation.
    static VEC_INIT: std::sync::Once = std::sync::Once::new();
    static VEC_INIT_RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

    fn init_sqlite_vec() -> PyResult<()> {
        VEC_INIT.call_once(|| {
            let result = unsafe {
                sqlite3_auto_extension(Some(std::mem::transmute::<
                    *const (),
                    unsafe extern "C" fn(
                        *mut rusqlite::ffi::sqlite3,
                        *mut *const c_char,
                        *const rusqlite::ffi::sqlite3_api_routines,
                    ) -> i32,
                >(sqlite3_vec_init as *const ())))
            };
            VEC_INIT_RESULT.store(result, std::sync::atomic::Ordering::Release);
        });
        if VEC_INIT_RESULT.load(std::sync::atomic::Ordering::Acquire) != rusqlite::ffi::SQLITE_OK {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "sqlite-vec registration failed",
            ));
        }
        Ok(())
    }

    #[pymodule]
    fn _kvault(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Initialize sqlite-vec extension before anything else
        init_sqlite_vec()?;

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
}
