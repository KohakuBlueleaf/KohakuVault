"""Tests for DataPacker vector/array support."""

import numpy as np
import pytest

from kohakuvault import ColumnVault, DataPacker


def test_vec_f32_arbitrary_shape():
    """Test arbitrary shape float32 vectors."""
    packer = DataPacker("vec:f32")

    # 1D array
    vec1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    packed = packer.pack(vec1d)
    unpacked = packer.unpack(packed, 0)

    assert isinstance(unpacked, np.ndarray)
    assert unpacked.dtype == np.float32
    assert unpacked.shape == (4,)
    np.testing.assert_array_almost_equal(unpacked, vec1d)


def test_vec_f32_2d_arbitrary():
    """Test 2D arbitrary shape arrays."""
    packer = DataPacker("vec:f32")

    # 2D array
    vec2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    packed = packer.pack(vec2d)
    unpacked = packer.unpack(packed, 0)

    assert isinstance(unpacked, np.ndarray)
    assert unpacked.shape == (3, 2)
    np.testing.assert_array_almost_equal(unpacked, vec2d)


def test_vec_f32_fixed_1d():
    """Test fixed shape 1D vectors (like embeddings)."""
    packer = DataPacker("vec:f32:128")

    # Create 128-dim embedding
    vec = np.random.randn(128).astype(np.float32)
    packed = packer.pack(vec)
    unpacked = packer.unpack(packed, 0)

    assert isinstance(unpacked, np.ndarray)
    assert unpacked.shape == (128,)
    np.testing.assert_array_almost_equal(unpacked, vec)

    # Check encoded size (1 byte type + 128*4 bytes data)
    assert len(packed) == 1 + 128 * 4
    assert packer.elem_size == 1 + 128 * 4


def test_vec_i64_fixed_2d():
    """Test fixed shape 2D int64 arrays."""
    packer = DataPacker("vec:i64:10:20")

    # Create 10x20 array
    vec = np.arange(200, dtype=np.int64).reshape(10, 20)
    packed = packer.pack(vec)
    unpacked = packer.unpack(packed, 0)

    assert isinstance(unpacked, np.ndarray)
    assert unpacked.shape == (10, 20)
    assert unpacked.dtype == np.int64
    np.testing.assert_array_equal(unpacked, vec)

    # Check encoded size (1 byte type + 200*8 bytes data)
    assert len(packed) == 1 + 200 * 8
    assert packer.elem_size == 1 + 200 * 8


def test_vec_f64_arbitrary():
    """Test float64 arbitrary shape."""
    packer = DataPacker("vec:f64")

    vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    packed = packer.pack(vec)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.dtype == np.float64
    np.testing.assert_array_almost_equal(unpacked, vec)


def test_vec_u8_fixed():
    """Test uint8 fixed shape (like images)."""
    packer = DataPacker("vec:u8:28:28")

    # Create 28x28 image
    image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    packed = packer.pack(image)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.dtype == np.uint8
    assert unpacked.shape == (28, 28)
    np.testing.assert_array_equal(unpacked, image)


def test_vec_from_python_list():
    """Test that Python lists are converted to numpy arrays."""
    packer = DataPacker("vec:f32")

    # Pass Python list instead of numpy array
    python_list = [1.0, 2.0, 3.0, 4.0]
    packed = packer.pack(python_list)
    unpacked = packer.unpack(packed, 0)

    assert isinstance(unpacked, np.ndarray)
    assert unpacked.shape == (4,)
    np.testing.assert_array_almost_equal(unpacked, python_list)


def test_vec_shape_validation():
    """Test that fixed shape validation works."""
    packer = DataPacker("vec:f32:4")

    # Correct shape
    vec_correct = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    packed = packer.pack(vec_correct)
    assert len(packed) == 1 + 4 * 4

    # Wrong shape should fail
    vec_wrong = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(Exception):  # Should raise shape mismatch
        packer.pack(vec_wrong)


def test_vec_different_dtypes():
    """Test all supported element types."""
    test_cases = [
        ("vec:f32:4", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
        ("vec:f64:4", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)),
        ("vec:i32:4", np.array([1, 2, 3, 4], dtype=np.int32)),
        ("vec:i64:4", np.array([1, 2, 3, 4], dtype=np.int64)),
        ("vec:u8:4", np.array([1, 2, 3, 4], dtype=np.uint8)),
        ("vec:u16:4", np.array([1, 2, 3, 4], dtype=np.uint16)),
        ("vec:u32:4", np.array([1, 2, 3, 4], dtype=np.uint32)),
        ("vec:u64:4", np.array([1, 2, 3, 4], dtype=np.uint64)),
    ]

    for dtype, vec in test_cases:
        packer = DataPacker(dtype)
        packed = packer.pack(vec)
        unpacked = packer.unpack(packed, 0)

        assert isinstance(unpacked, np.ndarray)
        np.testing.assert_array_equal(unpacked, vec)


def test_vec_arbitrary_vs_fixed_overhead():
    """Test overhead difference between arbitrary and fixed shape."""
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # Arbitrary shape: |type(1)|ndim(1)|shape(1*4)|data(4*4)|
    packer_arb = DataPacker("vec:f32")
    packed_arb = packer_arb.pack(vec)
    # 1 + 1 + 4 + 16 = 22 bytes
    assert len(packed_arb) == 22

    # Fixed shape: |type(1)|data(4*4)|
    packer_fixed = DataPacker("vec:f32:4")
    packed_fixed = packer_fixed.pack(vec)
    # 1 + 16 = 17 bytes
    assert len(packed_fixed) == 17

    # Both should unpack to same array
    np.testing.assert_array_almost_equal(
        packer_arb.unpack(packed_arb, 0), packer_fixed.unpack(packed_fixed, 0)
    )


def test_vec_is_varsize():
    """Test that arbitrary shape is variable-size, fixed is not."""
    packer_arb = DataPacker("vec:f32")
    packer_fixed = DataPacker("vec:f32:128")

    assert packer_arb.is_varsize is True
    assert packer_arb.elem_size == 0

    assert packer_fixed.is_varsize is False
    assert packer_fixed.elem_size == 1 + 128 * 4


def test_vec_3d_array():
    """Test 3D arrays (like RGB images or video frames)."""
    packer = DataPacker("vec:u8")

    # 3D array: 3 channels, 10x10 image
    image = np.random.randint(0, 256, size=(3, 10, 10), dtype=np.uint8)
    packed = packer.pack(image)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.shape == (3, 10, 10)
    np.testing.assert_array_equal(unpacked, image)


def test_vec_pack_many_fixed():
    """Test pack_many with fixed-shape vectors."""
    packer = DataPacker("vec:f32:4")

    # Create list of vectors
    vectors = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
    ]

    # Pack many
    packed_all = packer.pack_many(vectors)

    # Should be 3 vectors * 17 bytes each
    assert len(packed_all) == 3 * 17

    # Unpack many
    unpacked_all = packer.unpack_many(packed_all, count=3)

    assert len(unpacked_all) == 3
    for i, unpacked in enumerate(unpacked_all):
        assert isinstance(unpacked, np.ndarray)
        np.testing.assert_array_almost_equal(unpacked, vectors[i])


def test_vec_in_column():
    """Test vector packer integration with ColumnVault."""
    cv = ColumnVault(":memory:")

    # Create column with fixed-shape float32 vectors
    col = cv.create_column("embeddings", "vec:f32:128")

    # Append vectors
    vec1 = np.random.randn(128).astype(np.float32)
    vec2 = np.random.randn(128).astype(np.float32)

    col.append(vec1)
    col.append(vec2)

    assert len(col) == 2

    # Read back
    retrieved1 = col[0]
    retrieved2 = col[1]

    assert isinstance(retrieved1, np.ndarray)
    assert isinstance(retrieved2, np.ndarray)
    np.testing.assert_array_almost_equal(retrieved1, vec1)
    np.testing.assert_array_almost_equal(retrieved2, vec2)


def test_vec_extend_in_column():
    """Test bulk extend with vectors."""
    cv = ColumnVault(":memory:")
    col = cv.create_column("vectors", "vec:f32:64")

    # Create batch of vectors
    vectors = [np.random.randn(64).astype(np.float32) for _ in range(100)]

    # Bulk extend
    col.extend(vectors)

    assert len(col) == 100

    # Verify random samples
    for idx in [0, 50, 99]:
        retrieved = col[idx]
        np.testing.assert_array_almost_equal(retrieved, vectors[idx])


def test_vec_slice_operations():
    """Test slice read/write with vectors."""
    cv = ColumnVault(":memory:")
    col = cv.create_column("vecs", "vec:i64:10")

    # Populate
    vectors = [np.arange(10, dtype=np.int64) + i * 10 for i in range(50)]
    col.extend(vectors)

    # Slice read
    batch = col[10:20]
    assert len(batch) == 10
    for i, vec in enumerate(batch):
        np.testing.assert_array_equal(vec, vectors[10 + i])

    # Slice write
    new_vectors = [np.arange(10, dtype=np.int64) + 1000 for _ in range(5)]
    col[5:10] = new_vectors

    # Verify
    for i in range(5):
        retrieved = col[5 + i]
        np.testing.assert_array_equal(retrieved, new_vectors[i])


def test_vec_auto_dtype_conversion():
    """Test that input arrays are converted to correct dtype."""
    packer = DataPacker("vec:f32:4")

    # Input as int array (should be converted to float32)
    vec_int = np.array([1, 2, 3, 4], dtype=np.int32)
    packed = packer.pack(vec_int)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.dtype == np.float32
    np.testing.assert_array_almost_equal(unpacked, [1.0, 2.0, 3.0, 4.0])

    # Input as float64 (should be converted to float32)
    vec_f64 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    packed = packer.pack(vec_f64)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.dtype == np.float32


def test_vec_large_embedding():
    """Test realistic embedding sizes (768, 1536, 3072 dims)."""
    for dims in [768, 1536, 3072]:
        packer = DataPacker(f"vec:f32:{dims}")

        vec = np.random.randn(dims).astype(np.float32)
        packed = packer.pack(vec)
        unpacked = packer.unpack(packed, 0)

        assert unpacked.shape == (dims,)
        assert len(packed) == 1 + dims * 4
        np.testing.assert_array_almost_equal(unpacked, vec)


def test_vec_nested_list_input():
    """Test that nested Python lists work for 2D arrays."""
    packer = DataPacker("vec:i32")

    # 2D list
    data = [[1, 2, 3], [4, 5, 6]]
    packed = packer.pack(data)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.shape == (2, 3)
    assert unpacked.dtype == np.int32
    np.testing.assert_array_equal(unpacked, data)


def test_vec_empty_array():
    """Test that empty arrays are handled."""
    packer = DataPacker("vec:f32")

    # Empty array
    vec = np.array([], dtype=np.float32)
    packed = packer.pack(vec)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.shape == (0,)


def test_vec_dtype_string_parsing():
    """Test dtype string parsing for vectors."""
    # Valid dtypes
    valid = [
        "vec:f32",
        "vec:f64",
        "vec:i32",
        "vec:i64",
        "vec:u8",
        "vec:f32:128",
        "vec:i64:10:20",
        "vec:u8:3:224:224",  # RGB image
    ]

    for dtype_str in valid:
        packer = DataPacker(dtype_str)
        assert packer is not None

    # Invalid dtypes
    invalid = [
        "vec",  # Missing element type
        "vec:",  # Empty element type
        "vec:unknown",  # Unknown element type
        "vec:f32:0",  # Zero dimension
    ]

    for dtype_str in invalid:
        with pytest.raises(Exception):
            DataPacker(dtype_str)


def test_vec_round_trip_various_shapes():
    """Test round-trip for various array shapes."""
    test_cases = [
        (np.array([1.0], dtype=np.float32), "vec:f32:1"),  # Scalar-like
        (np.random.randn(100).astype(np.float32), "vec:f32:100"),  # Vector
        (np.arange(100, dtype=np.int64).reshape(10, 10), "vec:i64:10:10"),  # Matrix
        (np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8), "vec:u8:3:64:64"),  # Image
    ]

    for vec, dtype in test_cases:
        packer = DataPacker(dtype)
        packed = packer.pack(vec)
        unpacked = packer.unpack(packed, 0)

        assert unpacked.shape == vec.shape
        np.testing.assert_array_equal(unpacked, vec)


def test_vec_arbitrary_format_details():
    """Test the exact binary format for arbitrary shape."""
    packer = DataPacker("vec:f32")

    vec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # Shape (2, 2)
    packed = packer.pack(vec)

    # Format: |type(1)|ndim(1)|dim0(4)|dim1(4)|data(16)|
    assert packed[0] == 0x01  # ElementType::F32
    assert packed[1] == 2  # ndim
    assert int.from_bytes(packed[2:6], "little") == 2  # dim0
    assert int.from_bytes(packed[6:10], "little") == 2  # dim1
    # Data starts at byte 10


def test_vec_fixed_format_details():
    """Test the exact binary format for fixed shape."""
    packer = DataPacker("vec:f32:4")

    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    packed = packer.pack(vec)

    # Format: |type(1)|data(16)|
    assert len(packed) == 17
    assert packed[0] == 0x01  # ElementType::F32
    # Data starts at byte 1


def test_vec_properties():
    """Test packer properties for vector types."""
    # Fixed shape
    packer_fixed = DataPacker("vec:f32:128")
    assert packer_fixed.elem_size == 1 + 128 * 4
    assert packer_fixed.is_varsize is False

    # Arbitrary shape
    packer_arb = DataPacker("vec:f64")
    assert packer_arb.elem_size == 0
    assert packer_arb.is_varsize is True


def test_vec_multi_dimensional_fixed():
    """Test multi-dimensional fixed shapes."""
    # 3D tensor
    packer = DataPacker("vec:f32:2:3:4")

    tensor = np.random.randn(2, 3, 4).astype(np.float32)
    packed = packer.pack(tensor)
    unpacked = packer.unpack(packed, 0)

    assert unpacked.shape == (2, 3, 4)
    np.testing.assert_array_almost_equal(unpacked, tensor)


def test_vec_large_batch_extend():
    """Test extending column with large batch of vectors."""
    cv = ColumnVault(":memory:")
    col = cv.create_column("large_vectors", "vec:f32:384")

    # Create 1000 vectors
    vectors = [np.random.randn(384).astype(np.float32) for _ in range(1000)]

    # Bulk extend
    col.extend(vectors)

    assert len(col) == 1000

    # Spot check
    for idx in [0, 100, 500, 999]:
        retrieved = col[idx]
        np.testing.assert_array_almost_equal(retrieved, vectors[idx])


def test_vec_column_with_cache():
    """Test vector columns with cache enabled."""
    cv = ColumnVault(":memory:")
    col = cv.create_column("cached_vectors", "vec:f32:256")

    # Enable cache
    col.enable_cache(cap_bytes=10 * 1024 * 1024, flush_threshold=5 * 1024 * 1024)

    # Add vectors
    vectors = [np.random.randn(256).astype(np.float32) for _ in range(100)]
    col.extend(vectors)

    # Flush
    col.flush_cache()

    # Verify
    for i in range(100):
        retrieved = col[i]
        np.testing.assert_array_almost_equal(retrieved, vectors[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
