"""
Tests for dynamic chunk size functionality.

Tests min/max chunk behavior with small chunk sizes to verify:
1. Chunks grow correctly from min to max
2. Addressing works correctly with max_chunk_bytes
3. All operations (read/write/insert/delete) work correctly
"""

import os
import tempfile
import pytest
from kohakuvault import ColumnVault


# Test parameters: small chunks to observe growth behavior
MIN_CHUNK = 16  # 16 bytes
MAX_CHUNK = 1024  # 1KB
MAX_CHUNK_VARSIZE = 4096  # 4KB for varsize (v0.4.0: elements must fit in max)


def test_fixed_column_append_with_dynamic_chunks():
    """Test appending to fixed-size column with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    # Create i64 column (8 bytes per element)
    cv.create_column("nums", "i64")
    col = cv["nums"]

    # Append enough data to trigger chunk growth (64KB = 8192 elements)
    # With 8-byte elements and 1KB max chunks, that's ~8 chunks
    n_elements = 8192
    for i in range(n_elements):
        col.append(i * 2)

    assert len(col) == n_elements

    # Verify random access works correctly
    assert col[0] == 0
    assert col[100] == 200
    assert col[1000] == 2000
    assert col[5000] == 10000
    assert col[-1] == (n_elements - 1) * 2
    assert col[-100] == (n_elements - 100) * 2


def test_fixed_column_bulk_extend_with_dynamic_chunks():
    """Test extending column with bulk data."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("data", "i64")
    col = cv["data"]

    # Extend with 64KB worth of data at once
    n_elements = 8192
    data = list(range(n_elements))
    col.extend(data)

    assert len(col) == n_elements

    # Verify all data is correct
    for i in [0, 1, 100, 500, 1000, 4000, 7000, 8191]:
        assert col[i] == i


def test_fixed_column_random_access_write():
    """Test random access writes with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("vals", "i64")
    col = cv["vals"]

    # Create initial data
    n_elements = 4096
    col.extend([0] * n_elements)

    # Random writes across different chunks
    test_indices = [0, 1, 10, 100, 500, 1000, 2000, 3000, 4095]
    for idx in test_indices:
        col[idx] = idx * 10

    # Verify writes
    for idx in test_indices:
        assert col[idx] == idx * 10

    # Verify other elements unchanged
    assert col[50] == 0
    assert col[1500] == 0


def test_fixed_column_insert_with_dynamic_chunks():
    """Test insert operation with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("items", "i64")
    col = cv["items"]

    # Create base data
    base_data = list(range(0, 1000, 10))  # [0, 10, 20, ..., 990]
    col.extend(base_data)

    # Insert in middle
    col.insert(50, 9999)
    assert col[50] == 9999
    assert col[49] == 490
    assert col[51] == 500

    # Insert at beginning
    col.insert(0, 8888)
    assert col[0] == 8888
    assert col[1] == 0

    # Insert at end
    original_len = len(col)
    col.insert(original_len, 7777)
    assert col[-1] == 7777


def test_fixed_column_delete_with_dynamic_chunks():
    """Test delete operation with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("items", "i64")
    col = cv["items"]

    # Create data
    col.extend(list(range(1000)))
    original_len = len(col)

    # Delete from middle
    del col[500]
    assert len(col) == original_len - 1
    assert col[499] == 499
    assert col[500] == 501  # Shifted

    # Delete from beginning
    del col[0]
    assert len(col) == original_len - 2
    assert col[0] == 1

    # Delete from end
    del col[-1]
    assert len(col) == original_len - 3


def test_fixed_column_iteration_with_dynamic_chunks():
    """Test iteration with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("seq", "i64")
    col = cv["seq"]

    # Add data across multiple chunks
    n_elements = 4096
    expected = [i * 3 for i in range(n_elements)]
    col.extend(expected)

    # Iterate and verify
    result = list(col)
    assert result == expected


def test_float_column_with_dynamic_chunks():
    """Test f64 column with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    cv.create_column("floats", "f64")
    col = cv["floats"]

    # Add floating point data
    n_elements = 4096
    for i in range(n_elements):
        col.append(i * 0.5)

    # Verify
    assert len(col) == n_elements
    assert abs(col[0] - 0.0) < 0.001
    assert abs(col[100] - 50.0) < 0.001
    assert abs(col[1000] - 500.0) < 0.001
    assert abs(col[-1] - (n_elements - 1) * 0.5) < 0.001


def test_fixed_bytes_column_with_dynamic_chunks():
    """Test fixed-size bytes column with dynamic chunking."""
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    # 10-byte fixed size
    cv.create_column("fixed", "bytes:10")
    col = cv["fixed"]

    # Add data (10 bytes per element, so 1KB = ~102 elements per max chunk)
    n_elements = 2000
    for i in range(n_elements):
        col.append(f"item{i:05d}".encode())

    assert len(col) == n_elements

    # Verify (remember padding to 10 bytes)
    assert col[0] == b"item00000\x00"
    assert col[999] == b"item00999\x00"
    assert col[-1] == b"item01999\x00"


def test_varsize_bytes_append_with_dynamic_chunks():
    """Test variable-size bytes column with dynamic chunking."""
    # v0.4.0: Use larger max for varsize
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK_VARSIZE)

    cv.create_column("strings", "bytes")
    col = cv["strings"]

    # Add variable-size data (few dozen KB total)
    strings = []
    for i in range(1000):
        # Varying sizes from 10 to 100 bytes
        size = 10 + (i % 90)
        s = f"string_{i:04d}_".encode() + b"x" * size
        strings.append(s)
        col.append(s)

    assert len(col) == 1000

    # Verify random access
    for i in [0, 1, 10, 100, 500, 999]:
        assert col[i] == strings[i]


def test_varsize_bytes_extend_with_dynamic_chunks():
    """Test extending variable-size bytes column."""
    # v0.4.0: Use larger max for varsize
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK_VARSIZE)

    cv.create_column("data", "bytes")
    col = cv["data"]

    # Create varied-size strings
    strings = []
    for i in range(2000):
        size = 5 + (i % 50)
        s = f"data{i:05d}".encode() + b"_" * size
        strings.append(s)

    col.extend(strings)

    assert len(col) == 2000

    # Verify random samples
    for i in [0, 100, 500, 1000, 1500, 1999]:
        assert col[i] == strings[i]


def test_varsize_bytes_iteration_with_dynamic_chunks():
    """Test iteration over variable-size bytes with dynamic chunking."""
    # v0.4.0: Use larger max for varsize
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK_VARSIZE)

    cv.create_column("items", "bytes")
    col = cv["items"]

    # Add data
    strings = [f"item_{i:04d}".encode() * (1 + i % 10) for i in range(1000)]
    col.extend(strings)

    # Iterate and verify
    result = list(col)
    assert result == strings


def test_varsize_bytes_empty_and_large():
    """Test variable-size bytes with empty strings and large strings."""
    # v0.4.0: Use larger max for varsize since elements must fit in max_chunk_bytes
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK_VARSIZE)

    cv.create_column("mixed", "bytes")
    col = cv["mixed"]

    # Mix of empty, small, and large strings
    # v0.4.0: Elements must fit in max_chunk_bytes (4KB)
    test_data = [
        b"",
        b"small",
        b"x" * 500,  # Medium
        b"",
        b"medium_size_string_here",
        b"y" * 2000,  # Large but fits in 4KB
        b"",
        b"tiny",
    ]

    col.extend(test_data)

    assert len(col) == len(test_data)

    for i, expected in enumerate(test_data):
        assert col[i] == expected


def test_multiple_columns_same_vault():
    """Test multiple columns in same vault with dynamic chunking."""
    # v0.4.0: Use larger max for varsize columns
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK_VARSIZE)

    # Create multiple columns
    cv.create_column("ints", "i64")
    cv.create_column("floats", "f64")
    cv.create_column("strings", "bytes")

    ints = cv["ints"]
    floats = cv["floats"]
    strings = cv["strings"]

    # Add data to each
    n = 1000
    ints.extend(list(range(n)))
    floats.extend([i * 0.1 for i in range(n)])
    strings.extend([f"str{i:04d}".encode() for i in range(n)])

    # Verify all columns
    assert len(ints) == n
    assert len(floats) == n
    assert len(strings) == n

    assert ints[500] == 500
    assert abs(floats[500] - 50.0) < 0.001
    assert strings[500] == b"str0500"


def test_persistence_with_dynamic_chunks():
    """Test that dynamic chunks persist correctly."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        # Create and populate
        cv1 = ColumnVault(db_path, min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        cv1.create_column("data", "i64")
        col1 = cv1["data"]

        test_data = list(range(5000))
        col1.extend(test_data)

        # Close and reopen
        del cv1

        cv2 = ColumnVault(db_path, min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col2 = cv2["data"]

        # Verify persisted data
        assert len(col2) == 5000
        assert list(col2) == test_data

    finally:
        # Cleanup
        for ext in ["", "-wal", "-shm"]:
            try:
                os.remove(f"{db_path}{ext}")
            except (FileNotFoundError, PermissionError):
                pass


def test_chunk_addressing_correctness():
    """
    Test that chunk addressing uses max_chunk_bytes correctly.

    This is the key test for the fix: with dynamic chunks, all addressing
    should assume max_chunk_bytes, even if actual chunks are smaller.
    """
    cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)

    # i64 = 8 bytes, max_chunk = 1024 bytes, so 128 elements per chunk
    cv.create_column("test", "i64")
    col = cv["test"]

    # Add exactly 3 chunks worth + a bit more
    elements_per_max_chunk = MAX_CHUNK // 8
    n_elements = elements_per_max_chunk * 3 + 50

    # Add data
    col.extend(list(range(n_elements)))

    # Test accessing elements in each chunk
    # Chunk 0: indices 0 to 127
    assert col[0] == 0
    assert col[127] == 127

    # Chunk 1: indices 128 to 255
    assert col[128] == 128
    assert col[255] == 255

    # Chunk 2: indices 256 to 383
    assert col[256] == 256
    assert col[383] == 383

    # Chunk 3: indices 384+
    assert col[384] == 384
    assert col[n_elements - 1] == n_elements - 1

    # Test writes across chunk boundaries
    col[127] = 9999
    col[128] = 8888
    assert col[127] == 9999
    assert col[128] == 8888


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
