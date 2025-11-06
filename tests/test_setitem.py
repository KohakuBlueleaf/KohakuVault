"""
Test suite for setitem operations (v0.4.2).

Tests both fixed-size and variable-size columns with slice setitem support.
"""

import random

import pytest
from kohakuvault import ColumnVault


class TestFixedSizeSliceSetitem:
    """Test slice setitem for fixed-size columns."""

    def test_basic_slice_setitem_i64(self, tmp_path):
        """Test basic slice setitem for i64 column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))
        col[2:5] = [20, 30, 40]

        assert list(col) == [0, 1, 20, 30, 40, 5, 6, 7, 8, 9]

    def test_basic_slice_setitem_f64(self, tmp_path):
        """Test basic slice setitem for f64 column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("floats", "f64")

        col.extend([float(i) for i in range(10)])
        col[1:4] = [10.5, 20.5, 30.5]

        assert col[1] == pytest.approx(10.5)
        assert col[2] == pytest.approx(20.5)
        assert col[3] == pytest.approx(30.5)

    def test_slice_setitem_negative_indices(self, tmp_path):
        """Test slice setitem with negative indices."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))
        col[-3:-1] = [100, 200]

        assert list(col) == [0, 1, 2, 3, 4, 5, 6, 100, 200, 9]

    def test_slice_setitem_empty_slice(self, tmp_path):
        """Test setitem with empty slice (no-op)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(5)))
        col[2:2] = []  # Empty slice

        assert list(col) == [0, 1, 2, 3, 4]

    def test_slice_setitem_entire_column(self, tmp_path):
        """Test setting entire column via slice."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))
        col[:] = [i * 10 for i in range(10)]

        assert list(col) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_slice_setitem_length_mismatch(self, tmp_path):
        """Test that length mismatch raises ValueError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))

        with pytest.raises(ValueError, match="length mismatch"):
            col[2:5] = [10, 20]  # Need 3, got 2

        with pytest.raises(ValueError, match="length mismatch"):
            col[2:5] = [10, 20, 30, 40]  # Need 3, got 4

    def test_slice_setitem_type_error(self, tmp_path):
        """Test that type mismatch raises error."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))

        with pytest.raises((TypeError, RuntimeError)):
            col[2:5] = [10, "invalid", 30]

    def test_slice_setitem_step_not_supported(self, tmp_path):
        """Test that step slicing raises ValueError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10)))

        with pytest.raises(ValueError, match="Step slicing not supported"):
            col[::2] = [10, 20, 30, 40, 50]

    def test_large_slice_setitem(self, tmp_path):
        """Test setting large slice of elements."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(10000)))
        new_values = [i * 100 for i in range(5000)]
        col[2000:7000] = new_values

        assert col[2000] == 0
        assert col[3000] == 100000
        assert col[6999] == 499900

    def test_slice_setitem_multi_chunk(self, tmp_path):
        """Test slice setitem spanning multiple chunks."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(5000)))
        col[1000:3000] = [i * 2 for i in range(2000)]

        assert col[1000] == 0
        assert col[2000] == 2000
        assert col[2999] == 3998


class TestFixedSizeSetitemEdgeCases:
    """Test edge cases for fixed-size setitem."""

    def test_setitem_single_still_works(self, tmp_path):
        """Verify single element setitem still works."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        col.extend(list(range(5)))
        col[2] = 100

        assert list(col) == [0, 1, 100, 3, 4]

    def test_slice_setitem_with_cache(self, tmp_path):
        """Test that slice setitem works with cache enabled."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        with col.cache():
            col.extend(list(range(100)))

        # Setitem after cache context
        col[10:20] = [i * 10 for i in range(10)]

        assert col[10] == 0
        assert col[15] == 50

    def test_slice_setitem_bytes_fixed(self, tmp_path):
        """Test slice setitem for fixed-size bytes column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("hashes", "bytes:32")

        col.extend([b"x" * 32 for _ in range(10)])
        col[2:5] = [b"A" * 32, b"B" * 32, b"C" * 32]

        assert col[2] == b"A" * 32
        assert col[3] == b"B" * 32
        assert col[4] == b"C" * 32


class TestVariableSizeSingleSetitem:
    """Test single element setitem for variable-size columns with size-aware logic."""

    def test_varsize_setitem_same_size(self, tmp_path):
        """Test replacing with exact same size (Case 1: new_size == old_size)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Add elements with known sizes
        col.extend([b"abc", b"defgh", b"ijk", b"lmnop"])

        # Get initial metadata (bytes_used should stay same for same-size replacement)
        initial_length = len(col)

        # Replace with same size
        col[1] = b"XXXXX"  # 5 bytes, same as "defgh"

        assert col[0] == b"abc"
        assert col[1] == b"XXXXX"  # Changed
        assert col[2] == b"ijk"
        assert col[3] == b"lmnop"
        assert len(col) == initial_length

    def test_varsize_setitem_smaller(self, tmp_path):
        """Test replacing with smaller size (Case 1: new_size < old_size)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Add elements
        col.extend([b"short", b"this_is_longer", b"end"])

        # Replace with smaller (should leave fragment, NOT update bytes_used!)
        col[1] = b"tiny"  # 4 bytes vs 14 bytes original

        assert col[0] == b"short"
        assert col[1] == b"tiny"  # Smaller
        assert col[2] == b"end"
        assert len(col) == 3

        # Verify other elements still work (fragments shouldn't break reading)
        assert col[0] == b"short"
        assert col[2] == b"end"

    def test_varsize_setitem_larger_fits(self, tmp_path):
        """Test replacing with larger size that fits in available space (Case 2)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Add elements with space between them
        col.extend([b"a", b"bb", b"ccc", b"dddd"])

        # Replace with larger (should fit and shift subsequent elements in chunk)
        col[1] = b"LONGER_TEXT"  # Much larger than "bb"

        assert col[0] == b"a"
        assert col[1] == b"LONGER_TEXT"  # Larger
        assert col[2] == b"ccc"  # Should still be readable
        assert col[3] == b"dddd"  # Should still be readable

    def test_varsize_setitem_larger_rebuild(self, tmp_path):
        """Test replacing with larger size requiring chunk rebuild (Case 3)."""
        vault = ColumnVault(str(tmp_path / "test.db"), min_chunk_bytes=512, max_chunk_bytes=2048)
        col = vault.create_column("data", "bytes")

        # Fill chunk nearly to capacity
        large_elements = [b"X" * 100 for _ in range(18)]  # ~1800 bytes total
        col.extend(large_elements)

        # Replace one with much larger (should trigger rebuild)
        col[5] = b"Y" * 300  # 300 bytes vs 100 bytes

        # Verify all elements readable after rebuild
        assert col[4] == b"X" * 100
        assert col[5] == b"Y" * 300  # Updated
        assert col[6] == b"X" * 100
        assert len(col) == 18

    def test_varsize_setitem_mixed_sizes(self, tmp_path):
        """Test with randomly sized elements."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        random.seed(42)

        # Create elements with random sizes
        original_data = [b"x" * random.randint(5, 50) for _ in range(50)]
        col.extend(original_data)

        # Update some with smaller, some with same, some with larger
        col[10] = b"smaller"  # Smaller than original
        col[20] = b"y" * len(original_data[20])  # Same size
        col[30] = b"z" * (len(original_data[30]) + 20)  # Larger

        # Verify all updates
        assert col[10] == b"smaller"
        assert col[20] == b"y" * len(original_data[20])
        assert col[30] == b"z" * (len(original_data[30]) + 20)

        # Verify others unchanged
        for i in [0, 5, 15, 25, 35, 45]:
            assert col[i] == original_data[i]

    def test_varsize_setitem_msgpack(self, tmp_path):
        """Test setitem with msgpack structured data."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("records", "msgpack")

        # Add structured data
        col.extend(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob", "extra": "data"},
                {"id": 3, "name": "Charlie"},
            ]
        )

        # Replace with different sizes
        col[1] = {"id": 2, "name": "Bobby"}  # Smaller (removed "extra")
        col[2] = {"id": 3, "name": "Charlie", "age": 30, "city": "NYC"}  # Larger

        assert col[1] == {"id": 2, "name": "Bobby"}
        assert col[2] == {"id": 3, "name": "Charlie", "age": 30, "city": "NYC"}

    def test_varsize_setitem_negative_index(self, tmp_path):
        """Test setitem with negative indices."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c", b"d", b"e"])

        col[-1] = b"LAST"
        col[-3] = b"THIRD_FROM_END"

        assert col[-1] == b"LAST"
        assert col[-3] == b"THIRD_FROM_END"
        assert col[2] == b"THIRD_FROM_END"
        assert col[4] == b"LAST"

    def test_varsize_setitem_index_out_of_bounds(self, tmp_path):
        """Test that out of bounds raises IndexError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c"])

        with pytest.raises(IndexError, match="out of range"):
            col[10] = b"invalid"

        with pytest.raises(IndexError, match="out of range"):
            col[-10] = b"invalid"

    def test_varsize_setitem_preserves_order(self, tmp_path):
        """Test that setitem preserves element ordering."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Create sequence
        data = [f"item_{i}".encode() for i in range(20)]
        col.extend(data)

        # Update multiple elements with varying sizes
        col[5] = b"TINY"
        col[10] = b"MEDIUM_SIZE_DATA"
        col[15] = b"X" * 100  # Large

        # Verify order preserved
        for i in range(20):
            if i == 5:
                assert col[i] == b"TINY"
            elif i == 10:
                assert col[i] == b"MEDIUM_SIZE_DATA"
            elif i == 15:
                assert col[i] == b"X" * 100
            else:
                assert col[i] == data[i]

    def test_varsize_setitem_chunk_rebuild_scenario(self, tmp_path):
        """Test scenario that triggers chunk rebuild (Case 3)."""
        vault = ColumnVault(str(tmp_path / "test.db"), min_chunk_bytes=256, max_chunk_bytes=1024)
        col = vault.create_column("data", "bytes")

        # Fill chunk to near capacity
        elements = []
        for i in range(15):
            if i % 3 == 0:
                elements.append(b"S" * 50)  # Small
            else:
                elements.append(b"M" * 30)  # Medium

        col.extend(elements)

        # Replace a small one with large (should trigger rebuild)
        col[3] = b"LARGE" * 40  # 200 bytes, replacing 30 bytes

        # Verify chunk rebuild worked
        assert col[3] == b"LARGE" * 40
        assert col[0] == b"S" * 50
        assert col[14] == b"M" * 30

        # Verify all elements still accessible
        for i in range(15):
            _ = col[i]  # Should not raise


class TestVariableSizeSliceSetitem:
    """Test slice setitem for variable-size columns."""

    def test_varsize_slice_setitem_basic(self, tmp_path):
        """Test basic slice setitem for bytes column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"bb", b"ccc", b"dddd", b"eeeee"])
        col[1:4] = [b"XX", b"YYY", b"ZZZZ"]

        assert col[0] == b"a"
        assert col[1] == b"XX"
        assert col[2] == b"YYY"
        assert col[3] == b"ZZZZ"
        assert col[4] == b"eeeee"

    def test_varsize_slice_setitem_direct_mode_smaller(self, tmp_path):
        """Test direct mode (total new ≤ total old) with smaller total."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Original: b"1" (1) + b"23" (2) + b"456" (3) + b"7890" (4) = 10 bytes
        col.extend([b"1", b"23", b"456", b"7890"])

        # New: b"11111" (5) + b"2" (1) = 6 bytes (smaller total, should use direct mode)
        col[2:4] = [b"11111", b"2"]

        assert col[0] == b"1"
        assert col[1] == b"23"
        assert col[2] == b"11111"
        assert col[3] == b"2"

    def test_varsize_slice_setitem_direct_mode_equal(self, tmp_path):
        """Test direct mode with equal total size."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"abc", b"de", b"fghij"])
        # Total: 3 + 2 + 5 = 10 bytes

        col[0:3] = [b"ABCDE", b"FG", b"HIJ"]
        # Total: 5 + 2 + 3 = 10 bytes (same)

        assert col[0] == b"ABCDE"
        assert col[1] == b"FG"
        assert col[2] == b"HIJ"

    def test_varsize_slice_setitem_rebuild_mode(self, tmp_path):
        """Test rebuild mode (total new > total old)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c", b"d"])
        # Total: 4 bytes

        col[1:3] = [b"LONGER", b"TEXT"]
        # Total: 6 + 4 = 10 bytes (larger, should rebuild)

        assert col[0] == b"a"
        assert col[1] == b"LONGER"
        assert col[2] == b"TEXT"
        assert col[3] == b"d"

    def test_varsize_slice_setitem_msgpack(self, tmp_path):
        """Test slice setitem with msgpack."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("records", "msgpack")

        col.extend(
            [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
                {"id": 3, "name": "C"},
            ]
        )

        col[0:2] = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ]

        assert col[0] == {"id": 1, "name": "Alice", "age": 30}
        assert col[1] == {"id": 2, "name": "Bob", "age": 25}
        assert col[2] == {"id": 3, "name": "C"}

    def test_varsize_slice_setitem_empty_slice(self, tmp_path):
        """Test setitem with empty slice."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c"])
        col[1:1] = []  # No-op

        assert list(col) == [b"a", b"b", b"c"]

    def test_varsize_slice_setitem_length_mismatch(self, tmp_path):
        """Test that length mismatch raises ValueError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c", b"d"])

        with pytest.raises(ValueError, match="length mismatch"):
            col[1:3] = [b"X"]  # Need 2, got 1

    def test_varsize_slice_setitem_negative_indices(self, tmp_path):
        """Test slice setitem with negative indices."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        col.extend([b"a", b"b", b"c", b"d", b"e"])
        col[-3:-1] = [b"XXX", b"YYY"]

        assert col[2] == b"XXX"
        assert col[3] == b"YYY"
        assert col[4] == b"e"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
