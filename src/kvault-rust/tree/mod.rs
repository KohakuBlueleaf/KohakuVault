// Allow PyO3-specific false positive warnings
#![allow(clippy::useless_conversion)]

//! CSB+Tree (Cache-Sensitive B+Tree) implementation
//!
//! This module provides a cache-optimized B+Tree variant that stores child nodes
//! contiguously in memory for better cache locality. The tree supports:
//! - Generic key and value types
//! - Range queries with efficient iteration
//! - Cache-aligned nodes for optimal CPU cache performance
//! - PyO3 bindings for Python integration

mod arena;
mod csbtree;
mod iterator;
mod node;
pub(crate) mod pyobject_key;
pub(crate) mod python;

// Re-export Python types
pub use python::register_tree_types;

#[cfg(test)]
mod tests;
