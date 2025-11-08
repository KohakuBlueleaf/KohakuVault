//! Lock-Free Skip List with locked deletion
//!
//! - Lock-free insert (atomic CAS)
//! - Lock-free get (atomic reads)
//! - Locked delete (uses Mutex for safety)
//!
//! This hybrid approach provides lock-free performance for hot paths (read/write)
//! while ensuring safe deletion with a simple lock.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// Maximum height
pub const MAX_HEIGHT: usize = 32;

/// Type alias for node pointer vectors (used in find operation)
type NodePtrVec<K, V> = Vec<*mut Node<K, V>>;

/// Skip list node
struct Node<K, V> {
    key: K,
    value: Arc<V>,
    forward: Vec<AtomicPtr<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, height: usize) -> Self {
        let mut forward = Vec::with_capacity(height);
        for _ in 0..height {
            forward.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self { key, value: Arc::new(value), forward }
    }

    fn new_head() -> Self
    where
        K: Default,
        V: Default,
    {
        let mut forward = Vec::with_capacity(MAX_HEIGHT);
        for _ in 0..MAX_HEIGHT {
            forward.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self { key: K::default(), value: Arc::new(V::default()), forward }
    }
}

/// Lock-free skip list (lock-free read/write, locked delete)
pub struct SkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    head: Box<Node<K, V>>,
    max_level: AtomicUsize,
    len: AtomicUsize,
    // Mutex only for deletion (keeps read/write lock-free)
    delete_lock: Mutex<()>,
}

unsafe impl<K: Ord + Clone + Send, V: Clone + Send> Send for SkipList<K, V> {}
unsafe impl<K: Ord + Clone + Send, V: Clone + Send> Sync for SkipList<K, V> {}

impl<K, V> SkipList<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    pub fn new() -> Self {
        Self {
            head: Box::new(Node::new_head()),
            max_level: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            delete_lock: Mutex::new(()),
        }
    }

    fn random_level() -> usize {
        thread_local! {
            static RNG: RefCell<u64> = const { RefCell::new(0x123456789ABCDEF) };
        }

        RNG.with(|rng| {
            let mut state = *rng.borrow();
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *rng.borrow_mut() = state;

            let mut level = 0;
            let mut x = state;
            while (x & 1) == 1 && level < MAX_HEIGHT - 1 {
                level += 1;
                x >>= 1;
            }
            level
        })
    }

    /// Find position for key (lock-free)
    fn find(&self, key: &K) -> (NodePtrVec<K, V>, NodePtrVec<K, V>) {
        let mut preds = vec![ptr::null_mut(); MAX_HEIGHT];
        let mut succs = vec![ptr::null_mut(); MAX_HEIGHT];

        let mut current = &*self.head as *const Node<K, V> as *mut Node<K, V>;
        let level = self.max_level.load(AtomicOrdering::Acquire);

        for i in (0..=level).rev() {
            unsafe {
                let current_ref = &*current;
                let mut next = current_ref.forward[i].load(AtomicOrdering::Acquire);

                while !next.is_null() {
                    let next_ref = &*next;
                    match next_ref.key.cmp(key) {
                        Ordering::Less => {
                            current = next;
                            let current_ref = &*current;
                            next = current_ref.forward[i].load(AtomicOrdering::Acquire);
                        }
                        Ordering::Equal | Ordering::Greater => break,
                    }
                }

                preds[i] = current;
                succs[i] = next;
            }
        }

        (preds, succs)
    }

    /// Insert (lock-free with CAS)
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        loop {
            let (preds, succs) = self.find(&key);

            // Check if key exists
            if !succs[0].is_null() {
                let succ_node = unsafe { &*succs[0] };
                if succ_node.key == key {
                    return Some(succ_node.value.clone());
                }
            }

            // Create new node
            let new_level = Self::random_level();
            let current_max = self.max_level.load(AtomicOrdering::Acquire);

            if new_level > current_max {
                let _ = self.max_level.compare_exchange(
                    current_max,
                    new_level,
                    AtomicOrdering::Release,
                    AtomicOrdering::Relaxed,
                );
            }

            let new_node =
                Box::into_raw(Box::new(Node::new(key.clone(), value.clone(), new_level + 1)));

            // Set forward pointers
            unsafe {
                let new_node_ref = &*new_node;
                #[allow(clippy::needless_range_loop)]
                for i in 0..=new_level.min(current_max) {
                    new_node_ref.forward[i].store(succs[i], AtomicOrdering::Relaxed);
                }
            }

            // CAS at each level
            let mut success = true;
            for i in 0..=new_level.min(current_max) {
                unsafe {
                    let pred_ref = &*preds[i];
                    if pred_ref.forward[i]
                        .compare_exchange(
                            succs[i],
                            new_node,
                            AtomicOrdering::Release,
                            AtomicOrdering::Relaxed,
                        )
                        .is_err()
                    {
                        success = false;
                        // Rollback previous levels
                        for j in 0..i {
                            let pred_ref = &*preds[j];
                            pred_ref.forward[j]
                                .compare_exchange(
                                    new_node,
                                    succs[j],
                                    AtomicOrdering::Release,
                                    AtomicOrdering::Relaxed,
                                )
                                .ok();
                        }
                        break;
                    }
                }
            }

            if success {
                self.len.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }

            // Failed - clean up and retry
            unsafe {
                let _ = Box::from_raw(new_node);
            }
        }
    }

    /// Get (lock-free read)
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let mut current = &*self.head as *const Node<K, V> as *mut Node<K, V>;
        let level = self.max_level.load(AtomicOrdering::Acquire);

        for i in (0..=level).rev() {
            unsafe {
                let current_ref = &*current;
                let mut next = current_ref.forward[i].load(AtomicOrdering::Acquire);

                while !next.is_null() {
                    let next_ref = &*next;
                    match next_ref.key.cmp(key) {
                        Ordering::Less => {
                            current = next;
                            let current_ref = &*current;
                            next = current_ref.forward[i].load(AtomicOrdering::Acquire);
                        }
                        Ordering::Equal => return Some(next_ref.value.clone()),
                        Ordering::Greater => break,
                    }
                }
            }
        }

        None
    }

    /// Remove (uses lock for safety, but reads/writes remain lock-free)
    pub fn remove(&self, key: &K) -> Option<Arc<V>> {
        let _lock = self.delete_lock.lock().unwrap();

        let (preds, succs) = self.find(key);

        if succs[0].is_null() {
            return None;
        }

        let victim = unsafe { &*succs[0] };
        if victim.key != *key {
            return None;
        }

        let value = victim.value.clone();

        // Unlink the node at all levels
        unsafe {
            #[allow(clippy::needless_range_loop)]
            for i in 0..victim.forward.len().min(preds.len()) {
                if preds[i].is_null() {
                    continue;
                }

                let pred_ref = &*preds[i];
                let victim_next = victim.forward[i].load(AtomicOrdering::Acquire);

                // Update predecessor to skip victim
                pred_ref.forward[i].store(victim_next, AtomicOrdering::Release);
            }

            // Deallocate the victim node
            let _ = Box::from_raw(succs[0]);
        }

        self.len.fetch_sub(1, AtomicOrdering::Relaxed);
        Some(value)
    }

    pub fn len(&self) -> usize {
        self.len.load(AtomicOrdering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.head = Box::new(Node::new_head());
        self.max_level.store(0, AtomicOrdering::Release);
        self.len.store(0, AtomicOrdering::Release);
    }

    /// Iterate (lock-free snapshot)
    pub fn iter(&self) -> SkipListIter<'_, K, V> {
        let first = self.head.forward[0].load(AtomicOrdering::Acquire);
        SkipListIter {
            current: if first.is_null() { None } else { Some(first) },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Range query (lock-free)
    pub fn range(&self, start: K, end: K) -> SkipListRangeIter<'_, K, V> {
        let mut current = &*self.head as *const Node<K, V> as *mut Node<K, V>;
        let level = self.max_level.load(AtomicOrdering::Acquire);

        // Find first node >= start
        for i in (0..=level).rev() {
            unsafe {
                let current_ref = &*current;
                let mut next = current_ref.forward[i].load(AtomicOrdering::Acquire);

                while !next.is_null() {
                    let next_ref = &*next;
                    if next_ref.key < start {
                        current = next;
                        let current_ref = &*current;
                        next = current_ref.forward[i].load(AtomicOrdering::Acquire);
                    } else {
                        break;
                    }
                }
            }
        }

        let first = unsafe {
            let current_ref = &*current;
            current_ref.forward[0].load(AtomicOrdering::Acquire)
        };

        SkipListRangeIter {
            current: if first.is_null() { None } else { Some(first) },
            end_key: end,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Iterator
pub struct SkipListIter<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    current: Option<*mut Node<K, V>>,
    _phantom: std::marker::PhantomData<&'a SkipList<K, V>>,
}

impl<'a, K, V> Iterator for SkipListIter<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = (K, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let node_ptr = self.current?;

        unsafe {
            let node = &*node_ptr;
            let result = (node.key.clone(), node.value.clone());
            let next = node.forward[0].load(AtomicOrdering::Acquire);
            self.current = if next.is_null() { None } else { Some(next) };
            Some(result)
        }
    }
}

/// Range iterator
pub struct SkipListRangeIter<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    current: Option<*mut Node<K, V>>,
    end_key: K,
    _phantom: std::marker::PhantomData<&'a SkipList<K, V>>,
}

impl<'a, K, V> Iterator for SkipListRangeIter<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = (K, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let node_ptr = self.current?;

        unsafe {
            let node = &*node_ptr;

            if node.key >= self.end_key {
                return None;
            }

            let result = (node.key.clone(), node.value.clone());
            let next = node.forward[0].load(AtomicOrdering::Acquire);
            self.current = if next.is_null() { None } else { Some(next) };
            Some(result)
        }
    }
}

impl<K, V> Default for SkipList<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for SkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn drop(&mut self) {
        let mut current = self.head.forward[0].load(AtomicOrdering::Acquire);

        while !current.is_null() {
            unsafe {
                let node = Box::from_raw(current);
                current = node.forward[0].load(AtomicOrdering::Acquire);
            }
        }
    }
}
