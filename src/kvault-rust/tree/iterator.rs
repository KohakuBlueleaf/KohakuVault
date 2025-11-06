//! Iterators for CSB+Tree traversal

use std::ops::{Bound, RangeBounds};

use super::arena::NodeArena;
use super::node::{NodeChildren, NodeId};

/// Iterator over all key-value pairs in sorted order
pub struct Iter<'a, K, V> {
    arena: &'a NodeArena<K, V>,
    current_node: Option<NodeId>,
    current_index: usize,
}

impl<'a, K, V> Iter<'a, K, V> {
    pub(crate) fn new(arena: &'a NodeArena<K, V>, leaf_head: Option<NodeId>) -> Self {
        Self { arena, current_node: leaf_head, current_index: 0 }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node_id = self.current_node?;
            let node = self.arena.get(node_id);

            if self.current_index < node.num_keys {
                let key = node.keys[self.current_index].as_ref()?;

                if let NodeChildren::Leaf { ref values, .. } = node.children {
                    let value = values[self.current_index].as_ref()?;
                    self.current_index += 1;
                    return Some((key, value));
                }
            } else {
                // Move to next leaf
                if let NodeChildren::Leaf { next, .. } = node.children {
                    self.current_node = next;
                    self.current_index = 0;
                } else {
                    return None;
                }
            }
        }
    }
}

/// Range iterator over keys in specified range
pub struct RangeIter<'a, K, V> {
    arena: &'a NodeArena<K, V>,
    current_node: Option<NodeId>,
    current_index: usize,
    start_bound: Bound<K>,
    end_bound: Bound<K>,
    started: bool,
}

impl<'a, K, V> RangeIter<'a, K, V>
where
    K: Ord + Clone,
{
    pub(crate) fn new<R>(
        arena: &'a NodeArena<K, V>,
        root: Option<NodeId>,
        leaf_head: Option<NodeId>,
        range: R,
    ) -> Self
    where
        R: RangeBounds<K>,
    {
        let start_bound = match range.start_bound() {
            Bound::Included(k) => Bound::Included(k.clone()),
            Bound::Excluded(k) => Bound::Excluded(k.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };

        let end_bound = match range.end_bound() {
            Bound::Included(k) => Bound::Included(k.clone()),
            Bound::Excluded(k) => Bound::Excluded(k.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };

        // Find starting leaf node
        let start_node = match start_bound {
            Bound::Included(ref k) | Bound::Excluded(ref k) => {
                root.map(|root_id| Self::find_leaf_for_key(arena, root_id, k))
            }
            Bound::Unbounded => leaf_head,
        };

        Self {
            arena,
            current_node: start_node,
            current_index: 0,
            start_bound,
            end_bound,
            started: false,
        }
    }

    fn find_leaf_for_key(arena: &NodeArena<K, V>, mut node_id: NodeId, key: &K) -> NodeId {
        loop {
            let node = arena.get(node_id);

            if node.is_leaf() {
                return node_id;
            }

            // Internal node: find child
            if let NodeChildren::Internal { ref children } = node.children {
                match node.find_key(key) {
                    Ok(pos) => {
                        node_id = children[pos + 1].expect("Child should exist");
                    }
                    Err(pos) => {
                        node_id = children[pos].expect("Child should exist");
                    }
                }
            }
        }
    }

    fn key_in_range(&self, key: &K) -> bool {
        let after_start = match &self.start_bound {
            Bound::Included(k) => key >= k,
            Bound::Excluded(k) => key > k,
            Bound::Unbounded => true,
        };

        let before_end = match &self.end_bound {
            Bound::Included(k) => key <= k,
            Bound::Excluded(k) => key < k,
            Bound::Unbounded => true,
        };

        after_start && before_end
    }
}

impl<'a, K, V> Iterator for RangeIter<'a, K, V>
where
    K: Ord + Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node_id = self.current_node?;
            let node = self.arena.get(node_id);

            // Find starting position in first node
            if !self.started {
                self.started = true;
                if let Bound::Included(ref k) | Bound::Excluded(ref k) = self.start_bound {
                    match node.find_key(k) {
                        Ok(pos) => {
                            if matches!(self.start_bound, Bound::Excluded(_)) {
                                self.current_index = pos + 1;
                            } else {
                                self.current_index = pos;
                            }
                        }
                        Err(pos) => {
                            self.current_index = pos;
                        }
                    }
                }
            }

            if self.current_index < node.num_keys {
                let key = node.keys[self.current_index].as_ref()?;

                // Check if key is in range
                if !self.key_in_range(key) {
                    return None;
                }

                if let NodeChildren::Leaf { ref values, .. } = node.children {
                    let value = values[self.current_index].as_ref()?;
                    self.current_index += 1;
                    return Some((key, value));
                }
            } else {
                // Move to next leaf
                if let NodeChildren::Leaf { next, .. } = node.children {
                    self.current_node = next;
                    self.current_index = 0;
                } else {
                    return None;
                }
            }
        }
    }
}
