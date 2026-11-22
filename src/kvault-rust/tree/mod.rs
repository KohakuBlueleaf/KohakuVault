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

pub mod arena;
pub mod csbtree;
pub mod iterator;
pub mod node;
#[cfg(feature = "python")]
pub(crate) mod pyobject_key;
#[cfg(feature = "python")]
pub(crate) mod python;

pub use csbtree::CSBPlusTree;
pub use iterator::{Iter, RangeIter};

// Re-export Python types
#[cfg(feature = "python")]
pub use python::register_tree_types;

#[cfg(test)]
mod tests;
