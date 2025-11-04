"""
Tests for variable-size column operations (v0.4.0).

Tests insert, delete, vacuum for data integrity with small chunks and random sizes.
"""

import pytest
import random
from kohakuvault import ColumnVault

# Small chunk sizes to stress-test chunking logic
MIN_CHUNK = 16  # 16 bytes
MAX_CHUNK = 512  # 512 bytes


class TestVarSizeExtend:
    """Test extend optimization with small chunks."""

    def test_extend_random_sizes(self):
        """Extend with random-length data (stress test chunking)."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "bytes")

        # Random lengths from 1 to 100 bytes
        random.seed(42)
        data = [b"x" * random.randint(1, 100) for _ in range(200)]

        col.extend(data)

        assert len(col) == 200
        # Verify data integrity
        for i in [0, 50, 100, 150, 199]:
            assert col[i] == data[i]

    def test_extend_msgpack_random(self):
        """Extend msgpack with varying sizes."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("records", "msgpack")

        random.seed(42)
        records = []
        for i in range(100):
            # Varying complexity
            record = {"id": i, "name": f"user_{i}"}
            if i % 3 == 0:
                record["extra"] = "x" * random.randint(10, 50)
            if i % 5 == 0:
                record["tags"] = [f"tag_{j}" for j in range(random.randint(1, 5))]
            records.append(record)

        col.extend(records)

        assert len(col) == 100
        # Verify random samples
        for i in [0, 25, 50, 75, 99]:
            assert col[i] == records[i]


class TestVarSizeDelete:
    """Test delete operation with small chunks."""

    def test_delete_random_data(self):
        """Delete from random-length data."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "bytes")

        random.seed(42)
        data = [b"x" * random.randint(5, 50) for _ in range(50)]
        col.extend(data)

        # Delete middle element
        del col[25]

        assert len(col) == 49
        assert col[24] == data[24]
        assert col[25] == data[26]  # Shifted
        assert col[48] == data[49]

    def test_delete_first(self):
        """Delete first element."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "bytes")

        col.extend([b"a", b"b" * 20, b"c" * 10])
        del col[0]

        assert len(col) == 2
        assert col[0] == b"b" * 20
        assert col[1] == b"c" * 10

    def test_delete_last(self):
        """Delete last element."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "bytes")

        col.extend([b"a" * 15, b"b" * 25, b"c" * 5])
        del col[-1]

        assert len(col) == 2
        assert col[0] == b"a" * 15
        assert col[1] == b"b" * 25

    def test_delete_multiple_random(self):
        """Delete multiple elements from random data."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "str:utf8")

        random.seed(42)
        strings = [f"str_{i}_" + "x" * random.randint(5, 30) for i in range(50)]
        col.extend(strings)

        original_len = len(col)

        # Delete several elements (from high to low to avoid index shifts)
        del col[40]
        del col[30]
        del col[20]
        del col[10]

        assert len(col) == original_len - 4

        # Verify remaining elements are correct (accounting for shifts)
        assert col[10] == strings[11]  # string_10 deleted, so strings[11] is now at index 10
        assert col[20] == strings[22]  # Two deletions before this
        assert col[30] == strings[33]  # Three deletions before this


class TestVarSizeDataIntegrity:
    """Test data integrity with small chunks and random data."""

    def test_append_extend_mix_random(self):
        """Mix append/extend with random-length data."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("data", "bytes")

        random.seed(42)
        all_data = []

        # Append some with random sizes
        for i in range(10):
            data = f"append_{i}_".encode() + b"x" * random.randint(5, 40)
            col.append(data)
            all_data.append(data)

        # Extend some with random sizes
        extend_data = [f"extend_{i}_".encode() + b"y" * random.randint(5, 40) for i in range(15)]
        col.extend(extend_data)
        all_data.extend(extend_data)

        # Append more with random sizes
        for i in range(10, 20):
            data = f"append_{i}_".encode() + b"z" * random.randint(5, 40)
            col.append(data)
            all_data.append(data)

        assert len(col) == 35  # 10 + 15 + 10
        # Verify all data matches
        for i in [0, 5, 9, 10, 15, 20, 24, 25, 30, 34]:
            assert col[i] == all_data[i]

    def test_msgpack_random_structures(self):
        """Msgpack with varying structure sizes."""
        cv = ColumnVault(":memory:", min_chunk_bytes=MIN_CHUNK, max_chunk_bytes=MAX_CHUNK)
        col = cv.create_column("records", "msgpack")

        random.seed(42)
        records = []

        # Mix of different structures with random complexity
        for i in range(30):
            record = {"type": "event", "id": i}

            # Random string field
            if i % 2 == 0:
                record["data"] = "x" * random.randint(10, 80)

            # Random list field
            if i % 3 == 0:
                record["tags"] = [f"tag_{j}" for j in range(random.randint(1, 10))]

            # Random nested dict
            if i % 5 == 0:
                record["meta"] = {"nested": "y" * random.randint(5, 30)}

            records.append(record)

        col.extend(records)

        assert len(col) == 30
        # Verify all records match
        for i in [0, 5, 10, 15, 20, 25, 29]:
            assert col[i] == records[i]
