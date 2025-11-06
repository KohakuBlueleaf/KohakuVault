//! Arena allocator for contiguous node storage (CSB+Tree key feature)

use super::node::{Node, NodeId};

/// Arena for allocating nodes contiguously in memory
/// This is the key to CSB+Tree's cache efficiency
pub struct NodeArena<K, V> {
    /// All nodes stored in a single Vec for contiguous memory
    nodes: Vec<Node<K, V>>,
    /// Free list for recycling deleted nodes
    free_list: Vec<NodeId>,
}

impl<K, V> NodeArena<K, V> {
    /// Create a new empty arena
    pub fn new() -> Self {
        Self { nodes: Vec::new(), free_list: Vec::new() }
    }

    /// Allocate a single node
    pub fn alloc(&mut self, node: Node<K, V>) -> NodeId {
        if let Some(id) = self.free_list.pop() {
            self.nodes[id.0] = node;
            id
        } else {
            let id = NodeId(self.nodes.len());
            self.nodes.push(node);
            id
        }
    }

    /// Get reference to node
    pub fn get(&self, id: NodeId) -> &Node<K, V> {
        &self.nodes[id.0]
    }

    /// Get mutable reference to node
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node<K, V> {
        &mut self.nodes[id.0]
    }

    /// Clear all nodes
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_list.clear();
    }
}
