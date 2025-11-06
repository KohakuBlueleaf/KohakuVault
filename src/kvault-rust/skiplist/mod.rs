//! Lock-Free Skip List implementation with Python object support
//!
//! This is a probabilistic data structure that maintains sorted order with O(log N) operations.
//! Uses atomic Compare-And-Swap (CAS) operations for lock-free concurrent access.
//!
//! Compared to CSB+Tree:
//! - Lock-free and thread-safe (can be shared across threads)
//! - Simpler implementation (no complex rebalancing)
//! - Better for concurrent multi-producer/multi-consumer scenarios
//! - Faster range queries (simpler traversal)
//!
//! Features:
//! - Lock-free insert using atomic CAS
//! - Lock-free get (pure reads)
//! - Thread-safe iteration (snapshot view)
//! - No deletion (would require epoch-based GC)

pub mod core;
pub mod python;

pub use python::register_skiplist_types;

#[cfg(test)]
mod tests;
