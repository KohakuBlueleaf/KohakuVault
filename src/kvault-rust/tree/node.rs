//! Node structure for CSB+Tree with cache-line alignment

use std::cmp::Ordering;

/// Node ID as index into arena
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub usize);

/// Cache-aligned node structure
#[repr(align(64))]
pub struct Node<K, V> {
    /// Keys stored in sorted order (Vec for dynamic sizing)
    pub keys: Vec<Option<K>>,
    /// Child pointers or values depending on node type
    pub children: NodeChildren<V>,
    /// Number of keys currently in node
    pub num_keys: usize,
    /// Maximum keys allowed in this node
    pub max_keys: usize,
    /// Parent node (for easier navigation)
    pub parent: Option<NodeId>,
}

/// Children can be either internal nodes or leaf values
pub enum NodeChildren<V> {
    /// Internal node: stores all child pointers (Vec for dynamic sizing)
    Internal { children: Vec<Option<NodeId>> },
    /// Leaf node: stores values and sibling pointers (Vec for dynamic sizing)
    Leaf {
        values: Vec<Option<V>>,
        next: Option<NodeId>, // Next leaf for iteration
        prev: Option<NodeId>, // Previous leaf for reverse iteration
    },
}

impl<K, V> Node<K, V> {
    /// Create a new empty leaf node with specified capacity
    pub fn new_leaf(max_keys: usize) -> Self {
        let mut keys = Vec::with_capacity(max_keys);
        keys.resize_with(max_keys, || None);

        let mut values = Vec::with_capacity(max_keys);
        values.resize_with(max_keys, || None);

        Self {
            keys,
            children: NodeChildren::Leaf { values, next: None, prev: None },
            num_keys: 0,
            max_keys,
            parent: None,
        }
    }

    /// Create a new empty internal node with specified capacity
    pub fn new_internal(max_keys: usize) -> Self {
        let mut keys = Vec::with_capacity(max_keys);
        keys.resize_with(max_keys, || None);

        let mut children = Vec::with_capacity(max_keys + 1);
        children.resize_with(max_keys + 1, || None);

        Self {
            keys,
            children: NodeChildren::Internal { children },
            num_keys: 0,
            max_keys,
            parent: None,
        }
    }

    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self.children, NodeChildren::Leaf { .. })
    }

    /// Check if this node is full
    pub fn is_full(&self) -> bool {
        self.num_keys >= self.max_keys
    }
}

impl<K: Ord, V> Node<K, V> {
    /// Search for key in this node
    pub fn find_key(&self, key: &K) -> Result<usize, usize> {
        let mut left = 0;
        let mut right = self.num_keys;

        while left < right {
            let mid = left + (right - left) / 2;
            match self.keys[mid].as_ref().unwrap().cmp(key) {
                Ordering::Less => left = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => right = mid,
            }
        }

        Err(left)
    }

    /// Insert key-value at position
    pub fn insert_at(&mut self, pos: usize, key: K, value: V) {
        // Shift keys right
        for i in (pos..self.num_keys).rev() {
            self.keys[i + 1] = self.keys[i].take();
        }
        self.keys[pos] = Some(key);

        // Shift values right if leaf
        if let NodeChildren::Leaf { ref mut values, .. } = self.children {
            for i in (pos..self.num_keys).rev() {
                values[i + 1] = values[i].take();
            }
            values[pos] = Some(value);
        }

        self.num_keys += 1;
    }

    /// Remove key at position
    pub fn remove_at(&mut self, pos: usize) -> (K, V) {
        let key = self.keys[pos].take().unwrap();

        // Get value if leaf
        let value = if let NodeChildren::Leaf { ref mut values, .. } = self.children {
            values[pos].take().unwrap()
        } else {
            panic!("Cannot remove from internal node");
        };

        // Shift keys left
        for i in pos..self.num_keys - 1 {
            self.keys[i] = self.keys[i + 1].take();
        }

        // Shift values left if leaf
        if let NodeChildren::Leaf { ref mut values, .. } = self.children {
            for i in pos..self.num_keys - 1 {
                values[i] = values[i + 1].take();
            }
        }

        self.num_keys -= 1;
        (key, value)
    }
}
