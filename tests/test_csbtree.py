"""Tests for CSB+Tree implementation"""

import dataclasses

import pytest
from kohakuvault import CSBTree


def test_import():
    """Test that CSBTree can be imported"""
    assert CSBTree is not None


def test_create_empty_tree():
    """Test creating an empty tree"""
    tree = CSBTree()
    assert len(tree) == 0
    assert tree.is_empty()


def test_insert_and_get():
    """Test basic insert and get operations"""
    tree = CSBTree()
    tree.insert(1, "one")
    tree.insert(2, "two")
    tree.insert(3, "three")

    assert tree.get(1) == "one"
    assert tree.get(2) == "two"
    assert tree.get(3) == "three"
    assert tree.get(4) is None
    assert len(tree) == 3


def test_insert_replace():
    """Test that insert replaces existing value"""
    tree = CSBTree()
    old_val = tree.insert(1, "one")
    assert old_val is None

    old_val = tree.insert(1, "ONE")
    assert old_val == "one"

    assert tree.get(1) == "ONE"
    assert len(tree) == 1


def test_contains():
    """Test __contains__ operator"""
    tree = CSBTree()
    tree.insert(1, "one")
    tree.insert(2, "two")

    assert 1 in tree
    assert 2 in tree
    assert 3 not in tree


def test_remove():
    """Test removing keys"""
    tree = CSBTree()
    tree.insert(1, "one")
    tree.insert(2, "two")
    tree.insert(3, "three")

    old_val = tree.remove(2)
    assert old_val == "two"
    assert len(tree) == 2
    assert 2 not in tree
    assert tree.get(2) is None


def test_remove_nonexistent():
    """Test removing non-existent key"""
    tree = CSBTree()
    tree.insert(1, "one")

    result = tree.remove(2)
    assert result is None
    assert len(tree) == 1


def test_clear():
    """Test clearing all entries"""
    tree = CSBTree()
    for i in range(10):
        tree.insert(i, f"value{i}")

    assert len(tree) == 10
    tree.clear()
    assert len(tree) == 0
    assert tree.is_empty()


def test_iteration_order():
    """Test that iteration is in sorted key order"""
    tree = CSBTree()
    keys = [5, 2, 8, 1, 9, 3, 7, 4, 6]

    for k in keys:
        tree.insert(k, f"value{k}")

    # Collect keys from iteration
    iter_keys = [k for k, v in tree]

    # Should be sorted
    assert iter_keys == sorted(keys)


def test_keys_values_items():
    """Test keys(), values(), items() methods"""
    tree = CSBTree()
    tree.insert(1, "one")
    tree.insert(2, "two")
    tree.insert(3, "three")

    keys = tree.keys()
    assert keys == [1, 2, 3]

    values = tree.values()
    assert values == ["one", "two", "three"]

    items = tree.items()
    assert items == [(1, "one"), (2, "two"), (3, "three")]


def test_range_query():
    """Test range queries"""
    tree = CSBTree()
    for i in range(20):
        tree.insert(i, f"value{i}")

    # Range [5, 15)
    results = tree.range(5, 15)
    keys = [k for k, v in results]
    assert keys == list(range(5, 15))


def test_range_query_empty():
    """Test range query with no results"""
    tree = CSBTree()
    for i in range(10):
        tree.insert(i, f"value{i}")

    results = tree.range(20, 30)
    assert len(results) == 0


def test_string_keys():
    """Test with string keys"""
    tree = CSBTree()
    tree.insert("apple", 1)
    tree.insert("banana", 2)
    tree.insert("cherry", 3)

    assert tree.get("banana") == 2

    keys = tree.keys()
    assert keys == ["apple", "banana", "cherry"]


def test_mixed_value_types():
    """Test with different value types"""
    tree = CSBTree()
    tree.insert(1, "string")
    tree.insert(2, 42)
    tree.insert(3, [1, 2, 3])
    tree.insert(4, {"key": "value"})

    assert tree.get(1) == "string"
    assert tree.get(2) == 42
    assert tree.get(3) == [1, 2, 3]
    assert tree.get(4) == {"key": "value"}


def test_get_default():
    """Test get_default method"""
    tree = CSBTree()
    tree.insert(1, "one")

    assert tree.get_default(1, "default") == "one"
    assert tree.get_default(2, "default") == "default"


def test_setdefault():
    """Test setdefault method"""
    tree = CSBTree()

    # Key doesn't exist
    result = tree.setdefault(1, "one")
    assert result == "one"
    assert tree.get(1) == "one"

    # Key exists
    result = tree.setdefault(1, "ONE")
    assert result == "one"  # Returns existing
    assert tree.get(1) == "one"  # Unchanged


def test_update():
    """Test update method"""
    tree = CSBTree()
    tree.insert(1, "one")

    updates = [(2, "two"), (3, "three"), (1, "ONE")]
    tree.update(updates)

    assert tree.get(1) == "ONE"
    assert tree.get(2) == "two"
    assert tree.get(3) == "three"
    assert len(tree) == 3


def test_large_tree():
    """Test with many entries"""
    tree = CSBTree()
    n = 1000

    # Insert
    for i in range(n):
        tree.insert(i, f"value{i}")

    assert len(tree) == n

    # Query
    for i in range(n):
        assert tree.get(i) == f"value{i}"

    # Range query
    results = tree.range(100, 200)
    assert len(results) == 100
    assert results[0][0] == 100
    assert results[-1][0] == 199


def test_bytes_keys():
    """Test with bytes keys"""
    tree = CSBTree()
    tree.insert(b"key1", "value1")
    tree.insert(b"key2", "value2")
    tree.insert(b"key3", "value3")

    assert tree.get(b"key2") == "value2"
    assert len(tree) == 3


def test_custom_comparable_objects():
    """Test with custom comparable objects"""

    @dataclasses.dataclass(order=True)
    class Person:
        age: int
        name: str

    tree = CSBTree()
    tree.insert(Person(30, "Alice"), "data1")
    tree.insert(Person(25, "Bob"), "data2")
    tree.insert(Person(35, "Charlie"), "data3")

    # Should be ordered by age first, then name
    keys = tree.keys()
    assert keys[0].age == 25
    assert keys[1].age == 30
    assert keys[2].age == 35

    # Get by key
    assert tree.get(Person(30, "Alice")) == "data1"


def test_repr():
    """Test string representation"""
    tree = CSBTree()
    assert "CSBTree" in repr(tree)
    assert "len=0" in repr(tree)

    tree.insert(1, "one")
    assert "len=1" in repr(tree)


def test_empty_tree_operations():
    """Test operations on empty tree"""
    tree = CSBTree()

    assert tree.get(1) is None
    assert tree.remove(1) is None
    assert 1 not in tree
    assert len(tree) == 0
    assert tree.is_empty()

    assert tree.keys() == []
    assert tree.values() == []
    assert tree.items() == []

    results = list(tree)
    assert results == []

    range_results = tree.range(0, 10)
    assert range_results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
