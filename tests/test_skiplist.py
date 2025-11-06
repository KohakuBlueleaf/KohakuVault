"""Tests for SkipList implementation"""

import pytest
from kohakuvault import SkipList


def test_import():
    """Test that SkipList can be imported"""
    assert SkipList is not None


def test_create_empty():
    """Test creating empty skiplist"""
    sl = SkipList()
    assert len(sl) == 0
    assert sl.is_empty()


def test_insert_and_get():
    """Test basic insert and get"""
    sl = SkipList()
    sl[1] = "one"
    sl[2] = "two"
    sl[3] = "three"

    assert sl[1] == "one"
    assert sl[2] == "two"
    assert sl[3] == "three"
    assert len(sl) == 3


def test_insert_replace():
    """Test inserting duplicate key"""
    sl = SkipList()
    old = sl.insert(1, "one")
    assert old is None

    # Duplicate insert returns existing value (doesn't replace)
    old = sl.insert(1, "ONE")
    assert old == "one"  # Returns existing
    assert sl[1] == "one"  # Value unchanged
    assert len(sl) == 1


def test_remove():
    """Test deletion (uses lock but reads/writes are lock-free)"""
    sl = SkipList()
    sl[1] = "one"
    sl[2] = "two"
    sl[3] = "three"

    old = sl.remove(2)
    assert old == "two"
    assert len(sl) == 2
    assert 2 not in sl


def test_contains():
    """Test __contains__"""
    sl = SkipList()
    sl[1] = "one"
    sl[2] = "two"

    assert 1 in sl
    assert 2 in sl
    assert 3 not in sl


def test_iteration_order():
    """Test iteration is sorted"""
    sl = SkipList()
    keys = [5, 2, 8, 1, 9, 3, 7, 4, 6]

    for k in keys:
        sl[k] = f"value_{k}"

    iter_keys = [k for k, v in sl]
    assert iter_keys == sorted(keys)


def test_keys_values_items():
    """Test keys(), values(), items()"""
    sl = SkipList()
    sl[1] = "one"
    sl[2] = "two"
    sl[3] = "three"

    keys = sl.keys()
    assert keys == [1, 2, 3]

    values = sl.values()
    assert values == ["one", "two", "three"]

    items = sl.items()
    assert items == [(1, "one"), (2, "two"), (3, "three")]


def test_range_query():
    """Test range queries"""
    sl = SkipList()
    for i in range(100):
        sl[i] = f"value_{i}"

    result = sl.range(25, 75)
    keys = [k for k, v in result]
    assert keys == list(range(25, 75))


def test_range_query_empty():
    """Test empty range"""
    sl = SkipList()
    for i in range(10):
        sl[i] = f"value_{i}"

    result = sl.range(20, 30)
    assert len(result) == 0


def test_large_skiplist():
    """Test with large dataset"""
    sl = SkipList()
    n = 10000

    # Insert
    for i in range(n):
        sl[i] = f"value_{i}"

    assert len(sl) == n

    # Query
    for i in range(0, n, 100):
        assert sl[i] == f"value_{i}"

    # Range
    result = sl.range(1000, 2000)
    assert len(result) == 1000


def test_string_keys():
    """Test with string keys"""
    sl = SkipList()
    sl["apple"] = 1
    sl["banana"] = 2
    sl["cherry"] = 3

    assert sl["banana"] == 2
    keys = sl.keys()
    assert keys == ["apple", "banana", "cherry"]


def test_clear():
    """Test clear operation"""
    sl = SkipList()
    for i in range(10):
        sl[i] = f"value_{i}"

    assert len(sl) == 10
    sl.clear()
    assert len(sl) == 0
    assert sl.is_empty()


def test_dict_interface():
    """Test dict-like interface"""
    sl = SkipList()

    # setitem
    sl[5] = "five"
    sl[10] = "ten"

    # getitem
    assert sl[5] == "five"

    # Contains
    assert 5 in sl

    # KeyError
    with pytest.raises(KeyError):
        _ = sl[999]


def test_repr():
    """Test string representation"""
    sl = SkipList()
    assert "SkipList" in repr(sl)
    assert "len=0" in repr(sl)

    sl[1] = "one"
    assert "len=1" in repr(sl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
