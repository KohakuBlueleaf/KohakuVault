//! Tests for CSB+Tree implementation

use super::*;

#[test]
fn test_new_tree() {
    let tree: CSBPlusTree<i64, String> = CSBPlusTree::new();
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
}

#[test]
fn test_insert_single() {
    let mut tree = CSBPlusTree::new();
    assert_eq!(tree.insert(1, "one".to_string()), None);
    assert_eq!(tree.len(), 1);
    assert!(!tree.is_empty());
}

#[test]
fn test_insert_and_get() {
    let mut tree = CSBPlusTree::new();
    tree.insert(1, "one".to_string());
    tree.insert(2, "two".to_string());
    tree.insert(3, "three".to_string());

    assert_eq!(tree.get(&1), Some(&"one".to_string()));
    assert_eq!(tree.get(&2), Some(&"two".to_string()));
    assert_eq!(tree.get(&3), Some(&"three".to_string()));
    assert_eq!(tree.get(&4), None);
}

#[test]
fn test_insert_replace() {
    let mut tree = CSBPlusTree::new();
    assert_eq!(tree.insert(1, "one".to_string()), None);
    assert_eq!(tree.insert(1, "ONE".to_string()), Some("one".to_string()));
    assert_eq!(tree.get(&1), Some(&"ONE".to_string()));
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_insert_multiple() {
    let mut tree = CSBPlusTree::new();
    for i in 0..100 {
        tree.insert(i, format!("value{}", i));
    }
    assert_eq!(tree.len(), 100);

    for i in 0..100 {
        assert_eq!(tree.get(&i), Some(&format!("value{}", i)));
    }
}

#[test]
fn test_insert_reverse_order() {
    let mut tree = CSBPlusTree::new();
    for i in (0..50).rev() {
        tree.insert(i, format!("value{}", i));
    }
    assert_eq!(tree.len(), 50);

    for i in 0..50 {
        assert_eq!(tree.get(&i), Some(&format!("value{}", i)));
    }
}

#[test]
fn test_remove() {
    let mut tree = CSBPlusTree::new();
    tree.insert(1, "one".to_string());
    tree.insert(2, "two".to_string());
    tree.insert(3, "three".to_string());

    assert_eq!(tree.remove(&2), Some("two".to_string()));
    assert_eq!(tree.len(), 2);
    assert_eq!(tree.get(&2), None);
    assert_eq!(tree.get(&1), Some(&"one".to_string()));
    assert_eq!(tree.get(&3), Some(&"three".to_string()));
}

#[test]
fn test_remove_nonexistent() {
    let mut tree = CSBPlusTree::new();
    tree.insert(1, "one".to_string());
    assert_eq!(tree.remove(&2), None);
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_clear() {
    let mut tree = CSBPlusTree::new();
    for i in 0..10 {
        tree.insert(i, format!("value{}", i));
    }
    assert_eq!(tree.len(), 10);

    tree.clear();
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
    assert_eq!(tree.get(&5), None);
}

#[test]
fn test_iteration_order() {
    let mut tree = CSBPlusTree::new();
    let keys = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];

    for &k in &keys {
        tree.insert(k, format!("value{}", k));
    }

    // Collect keys from iteration
    let iter_keys: Vec<_> = tree.iter().map(|(k, _)| *k).collect();

    // Should be in sorted order
    let mut sorted_keys = keys.clone();
    sorted_keys.sort();
    assert_eq!(iter_keys, sorted_keys);
}

#[test]
fn test_iteration_values() {
    let mut tree = CSBPlusTree::new();
    tree.insert(1, "one".to_string());
    tree.insert(2, "two".to_string());
    tree.insert(3, "three".to_string());

    let items: Vec<_> = tree.iter().map(|(k, v)| (*k, v.clone())).collect();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0], (1, "one".to_string()));
    assert_eq!(items[1], (2, "two".to_string()));
    assert_eq!(items[2], (3, "three".to_string()));
}

#[test]
fn test_range_query_full() {
    let mut tree = CSBPlusTree::new();
    for i in 0..10 {
        tree.insert(i, format!("value{}", i));
    }

    let results: Vec<_> = tree.range(0..10).map(|(k, _)| *k).collect();
    assert_eq!(results, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_range_query_partial() {
    let mut tree = CSBPlusTree::new();
    for i in 0..20 {
        tree.insert(i, format!("value{}", i));
    }

    let results: Vec<_> = tree.range(5..15).map(|(k, _)| *k).collect();
    assert_eq!(results, vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
}

#[test]
fn test_range_query_empty() {
    let mut tree = CSBPlusTree::new();
    for i in 0..10 {
        tree.insert(i, format!("value{}", i));
    }

    let results: Vec<_> = tree.range(20..30).map(|(k, _)| *k).collect();
    assert_eq!(results, vec![]);
}

#[test]
fn test_range_query_single() {
    let mut tree = CSBPlusTree::new();
    for i in 0..10 {
        tree.insert(i, format!("value{}", i));
    }

    let results: Vec<_> = tree.range(5..6).map(|(k, _)| *k).collect();
    assert_eq!(results, vec![5]);
}

#[test]
fn test_string_keys() {
    let mut tree = CSBPlusTree::new();
    tree.insert("apple".to_string(), 1);
    tree.insert("banana".to_string(), 2);
    tree.insert("cherry".to_string(), 3);

    assert_eq!(tree.get(&"banana".to_string()), Some(&2));

    let keys: Vec<_> = tree.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(
        keys,
        vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string()
        ]
    );
}

#[test]
fn test_bytes_keys() {
    let mut tree = CSBPlusTree::new();
    tree.insert(vec![1, 2, 3], "first".to_string());
    tree.insert(vec![4, 5, 6], "second".to_string());
    tree.insert(vec![7, 8, 9], "third".to_string());

    assert_eq!(tree.get(&vec![4, 5, 6]), Some(&"second".to_string()));
    assert_eq!(tree.len(), 3);
}

#[test]
fn test_split_node() {
    // Insert enough keys to force node splits (MAX_KEYS = 7)
    let mut tree = CSBPlusTree::new();
    for i in 0..20 {
        tree.insert(i, format!("value{}", i));
    }

    // Verify all keys are still accessible after splits
    for i in 0..20 {
        assert_eq!(tree.get(&i), Some(&format!("value{}", i)));
    }
}

#[test]
fn test_large_tree() {
    let mut tree = CSBPlusTree::new();
    let n = 1000;

    // Insert
    for i in 0..n {
        tree.insert(i, format!("value{}", i));
    }
    assert_eq!(tree.len(), n);

    // Query
    for i in 0..n {
        assert_eq!(tree.get(&i), Some(&format!("value{}", i)));
    }

    // Range query
    let results: Vec<_> = tree.range(100..200).map(|(k, _)| *k).collect();
    assert_eq!(results.len(), 100);
    assert_eq!(results[0], 100);
    assert_eq!(results[99], 199);

    // Remove half
    for i in (0..n).step_by(2) {
        tree.remove(&i);
    }
    assert_eq!(tree.len(), n / 2);

    // Verify remaining
    for i in (1..n).step_by(2) {
        assert_eq!(tree.get(&i), Some(&format!("value{}", i)));
    }
}

#[test]
fn test_empty_tree_operations() {
    let mut tree: CSBPlusTree<i64, String> = CSBPlusTree::new();

    assert_eq!(tree.get(&1), None);
    assert_eq!(tree.remove(&1), None);

    let results: Vec<_> = tree.iter().collect();
    assert_eq!(results.len(), 0);

    let range_results: Vec<_> = tree.range(0..10).collect();
    assert_eq!(range_results.len(), 0);
}
