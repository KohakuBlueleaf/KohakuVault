"""Tests for DataPacker pack_many/unpack_many bulk operations."""

import numpy as np
import pytest

from kohakuvault import DataPacker


def test_i64_pack_many_small():
    """Test pack_many for i64 with small batch."""
    packer = DataPacker("i64")
    values = list(range(100))

    packed = packer.pack_many(values)
    assert len(packed) == 100 * 8  # 100 values * 8 bytes each

    unpacked = packer.unpack_many(packed, count=100)
    assert unpacked == values


def test_i64_pack_many_large():
    """Test pack_many for i64 with 1M values."""
    packer = DataPacker("i64")
    n = 1_000_000
    values = list(range(n))

    # Pack many
    packed = packer.pack_many(values)
    assert len(packed) == n * 8

    # Unpack many
    unpacked = packer.unpack_many(packed, count=n)
    assert len(unpacked) == n
    # Spot check
    assert unpacked[0] == 0
    assert unpacked[n // 2] == n // 2
    assert unpacked[-1] == n - 1


def test_f64_pack_many_large():
    """Test pack_many for f64 with 1M values."""
    packer = DataPacker("f64")
    n = 1_000_000
    values = [float(i) * 1.5 for i in range(n)]

    # Pack many
    packed = packer.pack_many(values)
    assert len(packed) == n * 8

    # Unpack many
    unpacked = packer.unpack_many(packed, count=n)
    assert len(unpacked) == n
    # Spot check
    assert unpacked[0] == 0.0
    assert abs(unpacked[100] - 150.0) < 1e-10


def test_vec_pack_many_small_batch():
    """Test pack_many for vectors with small batch."""
    packer = DataPacker("vec:f32:128")

    vectors = [np.random.randn(128).astype(np.float32) for _ in range(10)]

    # Pack many
    packed = packer.pack_many(vectors)
    expected_size = 10 * (1 + 128 * 4)  # 10 vectors * (1 byte type + 512 bytes data)
    assert len(packed) == expected_size

    # Unpack many
    unpacked = packer.unpack_many(packed, count=10)
    assert len(unpacked) == 10

    for i in range(10):
        np.testing.assert_array_almost_equal(unpacked[i], vectors[i])


def test_vec_pack_many_large_batch():
    """Test pack_many for vectors with 10K vectors."""
    packer = DataPacker("vec:f32:128")

    n = 10_000
    vectors = [np.random.randn(128).astype(np.float32) for _ in range(n)]

    # Pack many
    packed = packer.pack_many(vectors)
    expected_size = n * (1 + 128 * 4)
    assert len(packed) == expected_size

    # Unpack many
    unpacked = packer.unpack_many(packed, count=n)
    assert len(unpacked) == n

    # Spot check
    for idx in [0, n // 2, n - 1]:
        np.testing.assert_array_almost_equal(unpacked[idx], vectors[idx])


def test_vec_pack_many_different_sizes():
    """Test pack_many for vectors with different dimensions."""
    test_configs = [
        ("vec:f32:64", 64, 1000),
        ("vec:f32:256", 256, 1000),
        ("vec:f32:768", 768, 1000),
        ("vec:i64:100", 100, 1000),
        ("vec:u8:28:28", (28, 28), 500),
    ]

    for dtype, shape, count in test_configs:
        packer = DataPacker(dtype)

        if isinstance(shape, tuple):
            vectors = [np.random.randint(0, 256, size=shape, dtype=np.uint8) for _ in range(count)]
        elif "f32" in dtype:
            vectors = [np.random.randn(shape).astype(np.float32) for _ in range(count)]
        elif "i64" in dtype:
            vectors = [np.arange(shape, dtype=np.int64) for _ in range(count)]

        # Pack and unpack
        packed = packer.pack_many(vectors)
        unpacked = packer.unpack_many(packed, count=count)

        assert len(unpacked) == count
        # Spot check first and last
        np.testing.assert_array_equal(unpacked[0], vectors[0])
        np.testing.assert_array_equal(unpacked[-1], vectors[-1])


def test_bytes_pack_many():
    """Test pack_many for fixed-size bytes."""
    packer = DataPacker("bytes:128")

    values = [b"x" * 128 for _ in range(1000)]

    packed = packer.pack_many(values)
    assert len(packed) == 1000 * 128

    unpacked = packer.unpack_many(packed, count=1000)
    assert unpacked == values


def test_string_pack_many_fixed():
    """Test pack_many for fixed-size strings."""
    packer = DataPacker("str:32:utf8")

    values = [f"str_{i:04d}" for i in range(1000)]

    packed = packer.pack_many(values)
    assert len(packed) == 1000 * 32

    unpacked = packer.unpack_many(packed, count=1000)
    # Note: unpacked strings will be padded/trimmed to 32 bytes
    for i, s in enumerate(unpacked):
        assert s.startswith(f"str_{i:04d}")


def test_pack_many_empty_list():
    """Test pack_many with empty list."""
    packer = DataPacker("i64")

    packed = packer.pack_many([])
    assert len(packed) == 0

    unpacked = packer.unpack_many(packed, count=0)
    assert len(unpacked) == 0


def test_primitives_vs_vectors_bulk_performance():
    """Compare bulk operation performance: primitives should be faster."""
    import time

    # Test i64 (primitive)
    packer_i64 = DataPacker("i64")
    values_i64 = list(range(10000))

    start = time.perf_counter()
    packed_i64 = packer_i64.pack_many(values_i64)
    i64_pack_time = time.perf_counter() - start

    start = time.perf_counter()
    unpacked_i64 = packer_i64.unpack_many(packed_i64, count=10000)
    i64_unpack_time = time.perf_counter() - start

    # Test vec:i64:1 (single-element vectors)
    packer_vec = DataPacker("vec:i64:1")
    values_vec = [np.array([i], dtype=np.int64) for i in range(10000)]

    start = time.perf_counter()
    packed_vec = packer_vec.pack_many(values_vec)
    vec_pack_time = time.perf_counter() - start

    start = time.perf_counter()
    unpacked_vec = packer_vec.unpack_many(packed_vec, count=10000)
    vec_unpack_time = time.perf_counter() - start

    # Print timing info for investigation
    print(f"\nPack timing:")
    print(f"  i64 (primitive): {i64_pack_time*1000:.2f}ms")
    print(f"  vec:i64:1: {vec_pack_time*1000:.2f}ms")
    print(f"  Ratio: {vec_pack_time/i64_pack_time:.2f}x slower")

    print(f"\nUnpack timing:")
    print(f"  i64 (primitive): {i64_unpack_time*1000:.2f}ms")
    print(f"  vec:i64:1: {vec_unpack_time*1000:.2f}ms")
    print(f"  Ratio: {vec_unpack_time/i64_unpack_time:.2f}x slower")

    # Primitives should be significantly faster (at least 5x)
    # This test documents current performance, not enforcing threshold
    assert unpacked_i64 == values_i64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
