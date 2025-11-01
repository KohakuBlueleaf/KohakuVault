"""Tests for columnar storage."""

import pytest
from kohakuvault import KVault, ColumnVault, NotFound


def test_columnar_basic():
    """Test basic columnar operations."""
    kv = KVault(":memory:")
    cv = ColumnVault(kv)

    # Create column
    cv.create_column("x", "i64")
    col = cv["x"]

    # Initially empty
    assert len(col) == 0

    # Append elements
    col.append(0)
    col.append(1)
    col.append(4)

    assert len(col) == 3
    assert col[0] == 0
    assert col[1] == 1
    assert col[2] == 4

    # Negative indexing
    assert col[-1] == 4
    assert col[-2] == 1


def test_columnar_types():
    """Test different column types."""
    cv = ColumnVault(":memory:")

    # i64
    cv.create_column("ints", "i64")
    ints = cv["ints"]
    ints.append(123)
    ints.append(-456)
    assert ints[0] == 123
    assert ints[1] == -456

    # f64
    cv.create_column("floats", "f64")
    floats = cv["floats"]
    floats.append(3.14)
    floats.append(-2.71)
    assert abs(floats[0] - 3.14) < 0.001
    assert abs(floats[1] - (-2.71)) < 0.001

    # bytes
    cv.create_column("data", "bytes:10")
    data = cv["data"]
    data.append(b"hello")
    data.append(b"world!")
    assert data[0] == b"hello\x00\x00\x00\x00\x00"  # Padded to 10 bytes
    assert data[1] == b"world!\x00\x00\x00\x00"


def test_columnar_iteration():
    """Test iterating over columns."""
    cv = ColumnVault(":memory:")
    cv.create_column("nums", "i64")
    col = cv["nums"]

    # Add some data
    for i in range(10):
        col.append(i * i)

    # Iterate
    result = list(col)
    expected = [i * i for i in range(10)]
    assert result == expected


def test_columnar_setitem():
    """Test setting elements."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    col.append(10)
    col.append(20)
    col.append(30)

    # Update elements
    col[0] = 100
    col[1] = 200
    col[-1] = 300

    assert col[0] == 100
    assert col[1] == 200
    assert col[2] == 300


def test_columnar_delitem():
    """Test deleting elements."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    for i in range(5):
        col.append(i)

    assert len(col) == 5

    # Delete from middle
    del col[2]  # Remove 2
    assert len(col) == 4
    assert list(col) == [0, 1, 3, 4]

    # Delete last
    del col[-1]
    assert len(col) == 3
    assert list(col) == [0, 1, 3]


def test_columnar_insert():
    """Test inserting elements."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    col.extend([1, 2, 3])
    assert list(col) == [1, 2, 3]

    # Insert at beginning
    col.insert(0, 0)
    assert list(col) == [0, 1, 2, 3]

    # Insert in middle
    col.insert(2, 99)
    assert list(col) == [0, 1, 99, 2, 3]

    # Insert at end (append)
    col.insert(5, 4)
    assert list(col) == [0, 1, 99, 2, 3, 4]


def test_columnar_extend():
    """Test extending with multiple values."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    col.extend([1, 2, 3])
    assert len(col) == 3

    col.extend([4, 5, 6, 7])
    assert len(col) == 7
    assert list(col) == [1, 2, 3, 4, 5, 6, 7]


def test_columnar_clear():
    """Test clearing a column."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    col.extend([1, 2, 3, 4, 5])
    assert len(col) == 5

    col.clear()
    assert len(col) == 0
    assert list(col) == []

    # Can still append after clear
    col.append(10)
    assert len(col) == 1
    assert col[0] == 10


def test_columnar_ensure():
    """Test ensure method (get or create)."""
    cv = ColumnVault(":memory:")

    # Create if not exists
    col1 = cv.ensure("x", "i64")
    col1.append(123)

    # Get existing
    col2 = cv.ensure("x", "i64")
    assert col2[0] == 123
    assert col1 is col2  # Same instance


def test_columnar_list_columns():
    """Test listing columns."""
    cv = ColumnVault(":memory:")

    assert cv.list_columns() == []

    cv.create_column("a", "i64")
    cv.create_column("b", "f64")
    cv.create_column("c", "bytes:10")

    cols = cv.list_columns()
    assert len(cols) == 3

    names = [name for name, _, _ in cols]
    assert "a" in names
    assert "b" in names
    assert "c" in names


def test_columnar_delete_column():
    """Test deleting a column."""
    cv = ColumnVault(":memory:")

    cv.create_column("x", "i64")
    col = cv["x"]
    col.extend([1, 2, 3])

    assert cv.delete_column("x") is True
    assert cv.delete_column("x") is False  # Already deleted

    # Can't access deleted column
    with pytest.raises(NotFound):
        _ = cv["x"]


def test_columnar_shared_database():
    """Test that columnar and KV can share same database."""
    kv = KVault("test_shared.db")
    cv = ColumnVault(kv)

    # Use both interfaces
    kv["key1"] = b"value1"
    cv.create_column("col1", "i64")
    cv["col1"].extend([1, 2, 3])

    # Both should persist
    kv2 = KVault("test_shared.db")
    cv2 = ColumnVault(kv2)

    assert kv2["key1"] == b"value1"
    assert list(cv2["col1"]) == [1, 2, 3]

    # Cleanup
    import os

    kv.close()
    kv2.close()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"test_shared.db{ext}")
        except FileNotFoundError:
            pass


def test_columnar_large_data():
    """Test with larger dataset."""
    cv = ColumnVault(":memory:")
    cv.create_column("data", "i64")
    col = cv["data"]

    # Add 10,000 elements
    n = 10000
    for i in range(n):
        col.append(i)

    assert len(col) == n
    assert col[0] == 0
    assert col[n // 2] == n // 2
    assert col[-1] == n - 1

    # Iterate and verify
    for i, val in enumerate(col):
        if i >= 100:  # Just check first 100 for speed
            break
        assert val == i


def test_columnar_out_of_bounds():
    """Test out of bounds access."""
    cv = ColumnVault(":memory:")
    cv.create_column("x", "i64")
    col = cv["x"]

    col.extend([1, 2, 3])

    with pytest.raises(IndexError):
        _ = col[10]

    with pytest.raises(IndexError):
        _ = col[-10]


def test_columnar_column_not_found():
    """Test accessing non-existent column."""
    cv = ColumnVault(":memory:")

    with pytest.raises(NotFound):
        _ = cv["missing"]
