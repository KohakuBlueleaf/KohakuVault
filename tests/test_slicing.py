"""
Test suite for efficient slice-based column reading (v0.4.2).

Tests both fixed-size and variable-size columns with batch reading.
"""

import pytest
from kohakuvault import ColumnVault


class TestFixedSizeSlicing:
    """Test slicing for fixed-size columns (i64, f64)."""

    def test_basic_slice_i64(self, tmp_path):
        """Test basic slicing for i64 column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        # Add 1000 elements
        data = list(range(1000))
        col.extend(data)

        # Test various slices
        assert col[10:20] == data[10:20]
        assert col[0:100] == data[0:100]
        assert col[900:1000] == data[900:1000]
        assert col[:50] == data[:50]
        assert col[950:] == data[950:]
        assert col[-10:] == data[-10:]
        assert col[-100:-50] == data[-100:-50]

    def test_basic_slice_f64(self, tmp_path):
        """Test basic slicing for f64 column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("floats", "f64")

        # Add 500 elements
        data = [float(i) * 1.5 for i in range(500)]
        col.extend(data)

        # Test slices
        assert col[10:20] == pytest.approx(data[10:20])
        assert col[0:100] == pytest.approx(data[0:100])
        assert col[:50] == pytest.approx(data[:50])

    def test_empty_slice(self, tmp_path):
        """Test empty slices."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")
        col.extend(list(range(100)))

        # Empty slices
        assert col[50:50] == []
        assert col[100:200] == []
        assert col[50:40] == []  # stop < start

    def test_slice_single_element(self, tmp_path):
        """Test slice with single element."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")
        col.extend(list(range(100)))

        assert col[42:43] == [42]
        assert col[0:1] == [0]
        assert col[99:100] == [99]

    def test_slice_entire_column(self, tmp_path):
        """Test slicing entire column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        data = list(range(200))
        col.extend(data)

        assert col[:] == data
        assert col[0:200] == data
        assert col[0 : len(col)] == data

    def test_slice_step_not_supported(self, tmp_path):
        """Test that step slicing raises ValueError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")
        col.extend(list(range(100)))

        with pytest.raises(ValueError, match="Step slicing not supported"):
            _ = col[0:100:2]

        with pytest.raises(ValueError, match="Step slicing not supported"):
            _ = col[::3]

    def test_large_slice(self, tmp_path):
        """Test slicing large number of elements."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        # Add 10,000 elements
        data = list(range(10000))
        col.extend(data)

        # Large slice
        result = col[1000:9000]
        assert len(result) == 8000
        assert result == data[1000:9000]

    def test_slice_multi_chunk(self, tmp_path):
        """Test slicing that spans multiple chunks."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        # Add enough data to span multiple chunks
        data = list(range(1000))
        col.extend(data)

        # This slice should span multiple chunks
        result = col[100:900]
        assert result == data[100:900]


class TestVariableSizeSlicing:
    """Test slicing for variable-size columns (bytes, msgpack)."""

    def test_basic_slice_bytes(self, tmp_path):
        """Test basic slicing for bytes column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Add variable-size elements
        data = [f"element_{i}".encode() * (i % 10 + 1) for i in range(500)]
        col.extend(data)

        # Test slices
        assert col[10:20] == data[10:20]
        assert col[0:50] == data[0:50]
        assert col[:25] == data[:25]
        assert col[-10:] == data[-10:]

    def test_basic_slice_msgpack(self, tmp_path):
        """Test basic slicing for msgpack column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("records", "msgpack")

        # Add structured data
        data = [{"id": i, "value": f"item_{i}", "score": i * 1.5} for i in range(300)]
        col.extend(data)

        # Test slices
        assert col[10:20] == data[10:20]
        assert col[0:50] == data[0:50]
        assert col[100:150] == data[100:150]

    def test_empty_slice_varsize(self, tmp_path):
        """Test empty slices for variable-size columns."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")
        col.extend([b"data" for _ in range(100)])

        assert col[50:50] == []
        assert col[100:200] == []

    def test_slice_step_not_supported_varsize(self, tmp_path):
        """Test that step slicing raises ValueError for variable-size."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")
        col.extend([b"data" for _ in range(100)])

        with pytest.raises(ValueError, match="Step slicing not supported"):
            _ = col[0:100:2]

    def test_large_slice_varsize(self, tmp_path):
        """Test large slice for variable-size column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Add 5000 variable-size elements
        data = [f"element_{i}".encode() * (i % 20 + 1) for i in range(5000)]
        col.extend(data)

        # Large slice
        result = col[500:4500]
        assert len(result) == 4000
        assert result == data[500:4500]

    def test_slice_different_sizes(self, tmp_path):
        """Test slicing with widely varying element sizes."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        # Mix of small and large elements
        data = []
        for i in range(200):
            if i % 20 == 0:
                data.append(b"x" * 10000)  # Large element
            else:
                data.append(b"small")

        col.extend(data)

        # Slice that includes both small and large elements
        result = col[15:25]
        assert result == data[15:25]


class TestIterationOptimization:
    """Test that iteration uses optimized batch reading."""

    def test_iteration_fixed_size(self, tmp_path):
        """Test iteration for fixed-size column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        data = list(range(5000))
        col.extend(data)

        # Iteration should use batch unpacking
        result = list(col)
        assert result == data

    def test_iteration_varsize(self, tmp_path):
        """Test iteration for variable-size column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("data", "bytes")

        data = [f"element_{i}".encode() for i in range(2000)]
        col.extend(data)

        # Iteration should use batch reading
        result = list(col)
        assert result == data

    def test_iteration_msgpack(self, tmp_path):
        """Test iteration for msgpack column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("records", "msgpack")

        data = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        col.extend(data)

        result = list(col)
        assert result == data


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_index_type(self, tmp_path):
        """Test that invalid index types raise TypeError."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")
        col.extend(list(range(100)))

        with pytest.raises(TypeError, match="indices must be integers or slices"):
            _ = col["invalid"]

        with pytest.raises(TypeError, match="indices must be integers or slices"):
            _ = col[1.5]

    def test_slice_empty_column(self, tmp_path):
        """Test slicing empty column."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        assert col[:] == []
        assert col[0:10] == []
        assert col[10:20] == []

    def test_negative_indices_in_slice(self, tmp_path):
        """Test negative indices in slices."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        data = list(range(100))
        col.extend(data)

        assert col[-50:-40] == data[-50:-40]
        assert col[-10:] == data[-10:]
        assert col[:-10] == data[:-10]
        assert col[-100:50] == data[-100:50]

    def test_slice_correctness_vs_single_access(self, tmp_path):
        """Verify slice results match single-element access."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        data = list(range(200))
        col.extend(data)

        # Compare slice vs individual access
        start, end = 50, 100
        slice_result = col[start:end]
        individual_result = [col[i] for i in range(start, end)]

        assert slice_result == individual_result


class TestPerformanceCharacteristics:
    """Tests to verify performance characteristics (not actual benchmarks)."""

    def test_slice_large_range(self, tmp_path):
        """Test that large slices complete successfully."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        # Add 50,000 elements
        data = list(range(50000))
        col.extend(data)

        # Slice should be fast (single Rust call)
        result = col[10000:40000]
        assert len(result) == 30000
        assert result[0] == 10000
        assert result[-1] == 39999

    def test_many_small_slices(self, tmp_path):
        """Test many small slices (each should be a single Rust call)."""
        vault = ColumnVault(str(tmp_path / "test.db"))
        col = vault.create_column("ints", "i64")

        data = list(range(10000))
        col.extend(data)

        # Many small slices
        for i in range(0, 10000, 100):
            result = col[i : i + 10]
            assert result == data[i : i + 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
