//! Tests for SkipList implementation

use super::core::SkipList;

#[test]
fn test_new_skiplist() {
    let list: SkipList<i64, String> = SkipList::new();
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
}

#[test]
fn test_insert_and_get() {
    let mut list = SkipList::new();
    list.insert(1, "one".to_string());
    list.insert(2, "two".to_string());
    list.insert(3, "three".to_string());

    assert_eq!(list.get(&1), Some(&"one".to_string()));
    assert_eq!(list.get(&2), Some(&"two".to_string()));
    assert_eq!(list.get(&3), Some(&"three".to_string()));
    assert_eq!(list.get(&4), None);
}

#[test]
fn test_insert_replace() {
    let mut list = SkipList::new();
    assert_eq!(list.insert(1, "one".to_string()), None);
    assert_eq!(list.insert(1, "ONE".to_string()), Some("one".to_string()));
    assert_eq!(list.get(&1), Some(&"ONE".to_string()));
    assert_eq!(list.len(), 1);
}

#[test]
fn test_remove() {
    let mut list = SkipList::new();
    list.insert(1, "one".to_string());
    list.insert(2, "two".to_string());
    list.insert(3, "three".to_string());

    assert_eq!(list.remove(&2), Some("two".to_string()));
    assert_eq!(list.len(), 2);
    assert_eq!(list.get(&2), None);
}

#[test]
fn test_iteration_order() {
    let mut list = SkipList::new();
    let keys = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];

    for &k in &keys {
        list.insert(k, format!("value{}", k));
    }

    // Collect keys from iteration
    let iter_keys: Vec<_> = list.iter().map(|(k, _)| *k).collect();

    // Should be in sorted order
    let mut sorted_keys = keys.clone();
    sorted_keys.sort();
    assert_eq!(iter_keys, sorted_keys);
}

#[test]
fn test_large_skiplist() {
    let mut list = SkipList::new();
    let n = 1000;

    // Insert
    for i in 0..n {
        list.insert(i, format!("value{}", i));
    }
    assert_eq!(list.len(), n);

    // Query
    for i in 0..n {
        assert_eq!(list.get(&i), Some(&format!("value{}", i)));
    }

    // Remove half
    for i in (0..n).step_by(2) {
        list.remove(&i);
    }
    assert_eq!(list.len(), n / 2);
}
