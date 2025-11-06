//! Core CSB+Tree implementation with cache-optimized node storage

use std::ops::RangeBounds;

use super::arena::NodeArena;
use super::node::{Node, NodeChildren, NodeId};

/// Cache-Sensitive B+Tree with generic key and value types
pub struct CSBPlusTree<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Root node ID
    pub(crate) root: Option<NodeId>,
    /// Arena for contiguous node storage
    pub(crate) arena: NodeArena<K, V>,
    /// First leaf node (for iteration)
    pub(crate) leaf_head: Option<NodeId>,
    /// Total number of key-value pairs
    len: usize,
    /// Maximum keys per node (order)
    pub(crate) max_keys: usize,
}

impl<K, V> CSBPlusTree<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Create a new empty CSB+Tree with specified order (max keys per node)
    ///
    /// For optimal performance with Python objects:
    /// - Use 64-127 for small datasets (< 10K items)
    /// - Use 128-255 for medium datasets (10K - 1M items)
    /// - Use 256-511 for large datasets (> 1M items)
    ///
    /// Larger nodes = shallower tree = fewer Python comparison calls
    pub fn new_with_order(max_keys: usize) -> Self {
        Self { root: None, arena: NodeArena::new(), leaf_head: None, len: 0, max_keys }
    }

    /// Create a new CSB+Tree with default order of 63
    /// (balanced performance across different dataset sizes)
    pub fn new() -> Self {
        Self::new_with_order(63)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key-value pair, returns old value if key existed
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.root.is_none() {
            // Create first leaf node
            let mut leaf = Node::new_leaf(self.max_keys);
            leaf.insert_at(0, key, value);
            let leaf_id = self.arena.alloc(leaf);
            self.root = Some(leaf_id);
            self.leaf_head = Some(leaf_id);
            self.len = 1;
            return None;
        }

        // Find leaf node for key
        let leaf_id = self.find_leaf(self.root.unwrap(), &key);

        // Check if key already exists
        let leaf = self.arena.get(leaf_id);
        match leaf.find_key(&key) {
            Ok(pos) => {
                // Key exists, replace value
                let leaf = self.arena.get_mut(leaf_id);
                if let NodeChildren::Leaf { ref mut values, .. } = leaf.children {
                    let old_value = values[pos].replace(value);
                    return old_value;
                }
                unreachable!();
            }
            Err(pos) => {
                // Key doesn't exist, insert new
                let leaf = self.arena.get_mut(leaf_id);

                if !leaf.is_full() {
                    // Leaf has space
                    leaf.insert_at(pos, key, value);
                    self.len += 1;
                    None
                } else {
                    // Leaf is full, need to split
                    self.split_leaf_and_insert(leaf_id, pos, key, value);
                    self.len += 1;
                    None
                }
            }
        }
    }

    /// Get value for a key
    pub fn get(&self, key: &K) -> Option<&V> {
        let root = self.root?;
        let leaf_id = self.find_leaf(root, key);
        let leaf = self.arena.get(leaf_id);

        match leaf.find_key(key) {
            Ok(pos) => {
                if let NodeChildren::Leaf { ref values, .. } = leaf.children {
                    values[pos].as_ref()
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    /// Remove a key-value pair, returns value if key existed
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let root = self.root?;
        let leaf_id = self.find_leaf(root, key);
        let leaf = self.arena.get(leaf_id);

        match leaf.find_key(key) {
            Ok(pos) => {
                let leaf = self.arena.get_mut(leaf_id);
                let (_k, v) = leaf.remove_at(pos);
                self.len -= 1;

                // Handle underflow if needed (simplified: just allow underflow)
                // Full implementation would merge/redistribute nodes

                Some(v)
            }
            Err(_) => None,
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.root = None;
        self.arena.clear();
        self.leaf_head = None;
        self.len = 0;
    }

    /// Find the leaf node that should contain the key
    fn find_leaf(&self, mut node_id: NodeId, key: &K) -> NodeId {
        loop {
            let node = self.arena.get(node_id);

            if node.is_leaf() {
                return node_id;
            }

            // Internal node: find child to descend to
            if let NodeChildren::Internal { ref children } = node.children {
                match node.find_key(key) {
                    Ok(pos) => {
                        // Key found, go to child at pos + 1
                        node_id = children[pos + 1].expect("Child should exist");
                    }
                    Err(pos) => {
                        // Key not found, go to child at pos
                        node_id = children[pos].expect("Child should exist");
                    }
                }
            }
        }
    }

    /// Split a full leaf node and insert key-value
    fn split_leaf_and_insert(&mut self, leaf_id: NodeId, pos: usize, key: K, value: V) {
        let max_keys = self.max_keys;
        let mid = max_keys / 2;

        // Create new leaf for right half
        let mut new_leaf = Node::new_leaf(max_keys);

        // Determine where to insert
        let insert_left = pos <= mid;

        // Move keys/values to new leaf
        {
            let old_leaf = self.arena.get_mut(leaf_id);

            if let NodeChildren::Leaf { ref mut values, ref next, .. } = old_leaf.children {
                // Copy right half to new leaf
                for (src_idx, dst_idx) in (mid..old_leaf.num_keys).zip(0..) {
                    if let Some(k) = old_leaf.keys[src_idx].take() {
                        if let Some(v) = values[src_idx].take() {
                            new_leaf.insert_at(dst_idx, k, v);
                        }
                    }
                }
                old_leaf.num_keys = mid;

                // Update sibling pointers
                if let NodeChildren::Leaf {
                    next: ref mut new_next, prev: ref mut new_prev, ..
                } = new_leaf.children
                {
                    *new_next = *next;
                    *new_prev = Some(leaf_id);
                }
            }
        }

        // Insert new key-value
        if insert_left {
            let old_leaf = self.arena.get_mut(leaf_id);
            old_leaf.insert_at(pos, key, value);
        } else {
            new_leaf.insert_at(pos - mid, key, value);
        }

        // Get split key (first key of new leaf)
        let split_key = new_leaf.keys[0].as_ref().unwrap().clone();

        // Allocate new leaf
        let new_leaf_id = self.arena.alloc(new_leaf);

        // Update sibling pointers
        {
            // Get the old leaf's next pointer first
            let old_next = {
                let old_leaf = self.arena.get(leaf_id);
                if let NodeChildren::Leaf { next, .. } = old_leaf.children {
                    next
                } else {
                    None
                }
            };

            // Update old leaf's next pointer
            {
                let old_leaf = self.arena.get_mut(leaf_id);
                if let NodeChildren::Leaf { ref mut next, .. } = old_leaf.children {
                    *next = Some(new_leaf_id);
                }
            }

            // Update next node's prev pointer if it exists
            if let Some(next_id) = old_next {
                let next_node = self.arena.get_mut(next_id);
                if let NodeChildren::Leaf { ref mut prev, .. } = next_node.children {
                    *prev = Some(new_leaf_id);
                }
            }
        }

        // Insert split key into parent
        self.insert_into_parent(leaf_id, split_key, new_leaf_id);
    }

    /// Split an internal node
    fn split_internal_node(&mut self, node_id: NodeId) {
        let max_keys = self.max_keys;
        let mid = max_keys / 2;

        // Create new internal node for right half
        let mut new_internal = Node::new_internal(max_keys);

        // Copy data to split
        let (split_key, old_children) = {
            let old_internal = self.arena.get_mut(node_id);

            // Get split key (at mid)
            let split_key = old_internal.keys[mid].take().unwrap();

            // Copy old children for later use
            let old_children_copy =
                if let NodeChildren::Internal { ref children } = old_internal.children {
                    children.clone()
                } else {
                    vec![]
                };

            // Move keys mid+1..MAX_KEYS to new node
            for (src_idx, dst_idx) in ((mid + 1)..old_internal.num_keys).zip(0..) {
                if let Some(k) = old_internal.keys[src_idx].take() {
                    new_internal.keys[dst_idx] = Some(k);
                    new_internal.num_keys += 1;
                }
            }

            old_internal.num_keys = mid;

            (split_key, old_children_copy)
        };

        // Set up children for new internal node
        if let NodeChildren::Internal { ref mut children } = new_internal.children {
            // Copy children from mid+1 onwards
            for i in 0..=new_internal.num_keys {
                children[i] = old_children[mid + 1 + i];
            }
        }

        // Clear moved children from old node
        {
            let old_internal = self.arena.get_mut(node_id);
            if let NodeChildren::Internal { ref mut children } = old_internal.children {
                for child in children.iter_mut().skip(mid + 1) {
                    *child = None;
                }
            }
        }

        // Allocate new internal node
        let new_internal_id = self.arena.alloc(new_internal);

        // Collect child IDs first
        let child_ids: Vec<NodeId> = {
            let new_internal = self.arena.get(new_internal_id);
            if let NodeChildren::Internal { ref children } = new_internal.children {
                (0..=new_internal.num_keys)
                    .filter_map(|i| children[i])
                    .collect()
            } else {
                vec![]
            }
        };

        // Update parent pointers for children
        for child_id in child_ids {
            self.arena.get_mut(child_id).parent = Some(new_internal_id);
        }

        // Insert split key into parent (recursive)
        self.insert_into_parent(node_id, split_key, new_internal_id);
    }

    /// Insert key and right child into parent
    fn insert_into_parent(&mut self, left_id: NodeId, key: K, right_id: NodeId) {
        let parent_opt = self.arena.get(left_id).parent;

        if parent_opt.is_none() {
            // Create new root with two children
            let mut new_root = Node::new_internal(self.max_keys);
            new_root.keys[0] = Some(key);
            new_root.num_keys = 1;

            // Set up children array
            if let NodeChildren::Internal { ref mut children } = new_root.children {
                children[0] = Some(left_id);
                children[1] = Some(right_id);
            }

            let new_root_id = self.arena.alloc(new_root);

            // Update children's parent pointers
            self.arena.get_mut(left_id).parent = Some(new_root_id);
            self.arena.get_mut(right_id).parent = Some(new_root_id);

            self.root = Some(new_root_id);
            return;
        }

        // Insert into existing parent
        let parent_id = parent_opt.unwrap();

        // Check if parent is full - if so, split it first
        let parent_full = self.arena.get(parent_id).is_full();

        if parent_full {
            // Split internal node (similar to leaf splitting)
            self.split_internal_node(parent_id);
            // After splitting, recursively try to insert into the (now non-full) parent
            self.insert_into_parent(left_id, key, right_id);
            return;
        }

        // Find insertion position first (before mutable borrow)
        let pos = {
            let parent = self.arena.get(parent_id);
            match parent.find_key(&key) {
                Ok(_) => unreachable!("Split key should not exist in parent"),
                Err(p) => p,
            }
        };

        // Now do the insertion
        let parent = self.arena.get_mut(parent_id);

        // Shift keys right
        for i in (pos..parent.num_keys).rev() {
            parent.keys[i + 1] = parent.keys[i].take();
        }
        parent.keys[pos] = Some(key);

        // Shift children right
        if let NodeChildren::Internal { ref mut children } = parent.children {
            for i in (pos + 1..=parent.num_keys).rev() {
                children[i + 1] = children[i].take();
            }
            children[pos + 1] = Some(right_id);
        }

        parent.num_keys += 1;

        // Update child's parent pointer
        self.arena.get_mut(right_id).parent = Some(parent_id);
    }

    /// Create an iterator over all key-value pairs in sorted order
    pub fn iter(&self) -> super::iterator::Iter<'_, K, V> {
        super::iterator::Iter::new(&self.arena, self.leaf_head)
    }

    /// Create a range iterator over keys in the specified range
    pub fn range<R>(&self, range: R) -> super::iterator::RangeIter<'_, K, V>
    where
        R: RangeBounds<K>,
    {
        super::iterator::RangeIter::new(&self.arena, self.root, self.leaf_head, range)
    }
}

impl<K, V> Default for CSBPlusTree<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}
