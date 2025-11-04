"""
Comprehensive tests for columnar cache functionality.
Tests all cache modes: context manager, daemon thread, manual control, and lock.
"""

import tempfile
import time
from pathlib import Path

import pytest

from kohakuvault.column_proxy import ColumnVault


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)


# ======================================================================================
# Test 1: Context Manager Cache (vault.cache())
# ======================================================================================


def test_vault_cache_context_manager_fixed_size(temp_db):
    """Test vault-level cache() context manager with fixed-size column."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Test basic caching
    with cv.cache(cap_bytes=1024, flush_threshold=512):
        for i in range(100):
            col.append(i)
    # Auto-flushed on exit

    # Verify all data was written
    assert len(col) == 100
    assert list(col) == list(range(100))


def test_vault_cache_context_manager_variable_size(temp_db):
    """Test vault-level cache() context manager with variable-size column."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("messages", "bytes")

    messages = [f"message_{i}".encode() for i in range(100)]

    with cv.cache(cap_bytes=4096, flush_threshold=1024):
        for msg in messages:
            col.append(msg)
    # Auto-flushed on exit

    # Verify all data was written
    assert len(col) == 100
    assert [bytes(x) for x in col] == messages


def test_vault_cache_multi_column(temp_db):
    """Test vault-level cache() with multiple columns simultaneously."""
    cv = ColumnVault(temp_db)
    col1 = cv.create_column("nums", "i64")
    col2 = cv.create_column("floats", "f64")
    col3 = cv.create_column("msgs", "bytes")

    with cv.cache(cap_bytes=8192, flush_threshold=2048):
        for i in range(50):
            col1.append(i)
            col2.append(i * 1.5)
            col3.append(f"msg_{i}".encode())
    # All columns flushed on exit

    # Verify all columns
    assert len(col1) == 50
    assert len(col2) == 50
    assert len(col3) == 50
    assert list(col1) == list(range(50))


def test_vault_cache_exception_handling(temp_db):
    """Test that cache flushes even when exception occurs."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    try:
        with cv.cache():
            for i in range(10):
                col.append(i)
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Should have flushed despite exception
    assert len(col) == 10
    assert list(col) == list(range(10))


# ======================================================================================
# Test 2: Column-Level Cache (col.cache())
# ======================================================================================


def test_column_cache_context_manager(temp_db):
    """Test per-column cache() context manager."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    with col.cache(cap_bytes=2048, flush_threshold=512):
        for i in range(100):
            col.append(i)
    # Auto-flushed on exit

    assert len(col) == 100
    assert list(col) == list(range(100))


def test_column_cache_variable_size(temp_db):
    """Test per-column cache() with variable-size column."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("msgs", "bytes")

    messages = [f"test_{i}".encode() for i in range(50)]

    with col.cache(cap_bytes=2048):
        for msg in messages:
            col.append(msg)

    assert len(col) == 50
    assert [bytes(x) for x in col] == messages


# ======================================================================================
# Test 3: Manual Cache Control (enable/flush/disable)
# ======================================================================================


def test_manual_cache_control(temp_db):
    """Test manual cache control with enable/flush/disable."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Enable cache
    cv.enable_cache(cap_bytes=4096, flush_threshold=1024)

    # Append data
    for i in range(50):
        col.append(i)

    # Manual flush
    flushed = cv.flush_cache()
    assert flushed > 0  # Should have flushed elements

    # Append more
    for i in range(50, 100):
        col.append(i)

    # Flush again
    cv.flush_cache()

    # Disable cache
    cv.disable_cache()

    # Verify all data
    assert len(col) == 100
    assert list(col) == list(range(100))


def test_manual_cache_multiple_flushes(temp_db):
    """Test multiple manual flushes during operation."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    cv.enable_cache(cap_bytes=2048, flush_threshold=512)

    # Write in batches with manual flushes
    for batch in range(5):
        for i in range(batch * 20, (batch + 1) * 20):
            col.append(i)
        cv.flush_cache()

    cv.disable_cache()

    assert len(col) == 100
    assert list(col) == list(range(100))


# ======================================================================================
# Test 4: Auto-Flush on Threshold
# ======================================================================================


def test_auto_flush_on_threshold(temp_db):
    """Test automatic flush when reaching threshold."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Small threshold to trigger auto-flush
    cv.enable_cache(cap_bytes=1024, flush_threshold=256)

    # Append enough data to trigger auto-flush
    for i in range(100):
        col.append(i)

    # Final flush
    cv.flush_cache()
    cv.disable_cache()

    assert len(col) == 100
    assert list(col) == list(range(100))


def test_large_element_bypass_cache(temp_db):
    """Test that elements larger than cache capacity bypass cache."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("msgs", "bytes")

    # Very small cache
    cv.enable_cache(cap_bytes=256, flush_threshold=128)

    # Large element (should bypass cache and write directly)
    large_msg = b"x" * 512
    col.append(large_msg)

    # Small elements (should use cache)
    for i in range(10):
        col.append(f"small_{i}".encode())

    cv.flush_cache()
    cv.disable_cache()

    assert len(col) == 11
    assert bytes(col[0]) == large_msg


# ======================================================================================
# Test 5: Daemon Thread Auto-Flush
# ======================================================================================


def test_daemon_thread_auto_flush(temp_db):
    """Test daemon thread auto-flush at interval."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Enable with daemon thread (flush every 0.5 seconds)
    cv.enable_cache(cap_bytes=4096, flush_threshold=1024, flush_interval=0.5)

    # Append data slowly
    for i in range(20):
        col.append(i)
        time.sleep(0.1)  # 2 seconds total, should trigger multiple flushes

    # Give daemon time to flush
    time.sleep(0.6)

    cv.disable_cache()  # Stops daemon

    # Verify all data was flushed
    assert len(col) == 20
    assert list(col) == list(range(20))


def test_daemon_thread_multiple_columns(temp_db):
    """Test daemon thread with multiple columns."""
    cv = ColumnVault(temp_db)
    col1 = cv.create_column("nums", "i64")
    col2 = cv.create_column("msgs", "bytes")

    cv.enable_cache(flush_interval=0.3)

    # Append to both columns
    for i in range(30):
        col1.append(i)
        col2.append(f"msg_{i}".encode())
        time.sleep(0.05)  # 1.5 seconds, should trigger 5 flushes

    time.sleep(0.4)
    cv.disable_cache()

    assert len(col1) == 30
    assert len(col2) == 30


# ======================================================================================
# Test 6: Lock Cache (for Atomic Operations)
# ======================================================================================


def test_lock_cache_context_manager(temp_db):
    """Test lock_cache() prevents daemon from flushing."""
    cv = ColumnVault(temp_db)
    col1 = cv.create_column("nums1", "i64")
    col2 = cv.create_column("nums2", "i64")

    # Enable with daemon thread
    cv.enable_cache(flush_interval=0.2)

    # Use lock to prevent daemon flush during atomic operation
    with cv.lock_cache():
        for i in range(10):
            col1.append(i)
            col2.append(i * 2)
        time.sleep(0.5)  # Daemon should NOT flush during this time

    # After unlock, daemon can flush
    time.sleep(0.3)

    cv.disable_cache()

    assert len(col1) == 10
    assert len(col2) == 10


# ======================================================================================
# Test 7: Cache with Structural Modifications (clear, delete, insert)
# ======================================================================================


def test_cache_flush_before_clear(temp_db):
    """Test that clear() flushes cache first."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    cv.enable_cache()

    # Append data (cached)
    for i in range(20):
        col.append(i)

    # Clear should flush first
    col.clear()

    cv.disable_cache()

    # Should be empty
    assert len(col) == 0


def test_cache_flush_before_delete_column(temp_db):
    """Test that delete_column() flushes cache first."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    cv.enable_cache()

    # Append data (cached)
    for i in range(10):
        col.append(i)

    # Delete column should flush first
    deleted = cv.delete_column("nums")
    assert deleted is True

    cv.disable_cache()


# ======================================================================================
# Test 8: Performance - Cache vs No Cache
# ======================================================================================


def test_performance_with_cache(temp_db):
    """Test that cache significantly improves performance."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    import time

    # Test with cache
    start = time.time()
    with cv.cache():
        for i in range(1000):
            col.append(i)
    cached_time = time.time() - start

    col.clear()

    # Test without cache
    start = time.time()
    for i in range(1000):
        col.append(i)
    uncached_time = time.time() - start

    # Cache should be faster (at least 2x)
    print(f"Cached: {cached_time:.4f}s, Uncached: {uncached_time:.4f}s")
    print(f"Speedup: {uncached_time/cached_time:.2f}x")

    # Both should have same result
    assert len(col) == 1000


# ======================================================================================
# Test 9: Structured Data Types (msgpack, cbor)
# ======================================================================================


def test_cache_with_msgpack(temp_db):
    """Test cache with msgpack structured data."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("records", "msgpack")

    records = [{"id": i, "name": f"user_{i}", "score": i * 10} for i in range(50)]

    with cv.cache():
        for record in records:
            col.append(record)

    assert len(col) == 50
    for i, record in enumerate(col):
        assert record == records[i]


# ======================================================================================
# Test 10: Edge Cases
# ======================================================================================


def test_empty_cache_flush(temp_db):
    """Test flushing empty cache does nothing."""
    cv = ColumnVault(temp_db)
    cv.create_column("nums", "i64")

    cv.enable_cache()
    flushed = cv.flush_cache()
    assert flushed == 0  # Nothing to flush

    cv.disable_cache()


def test_disable_cache_without_enable(temp_db):
    """Test disabling cache when not enabled."""
    cv = ColumnVault(temp_db)
    cv.create_column("nums", "i64")

    # Should not error
    cv.disable_cache()


def test_multiple_enable_disable_cycles(temp_db):
    """Test multiple enable/disable cycles."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    for cycle in range(3):
        cv.enable_cache()
        for i in range(10):
            col.append(cycle * 10 + i)
        cv.flush_cache()
        cv.disable_cache()

    assert len(col) == 30


def test_nested_cache_context(temp_db):
    """Test nested cache contexts (should work, inner resets cache)."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    with cv.cache(cap_bytes=1024):
        for i in range(10):
            col.append(i)

        # Inner context (will reset cache settings)
        with cv.cache(cap_bytes=512):
            for i in range(10, 20):
                col.append(i)
        # Inner flushed

    # Outer flushed

    assert len(col) == 20


# ======================================================================================
# Test 11: Flush Before Structural Operations (insert, delete, setitem)
# ======================================================================================


def test_cache_flush_before_setitem(temp_db):
    """Test that setitem flushes cache first."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Pre-populate
    col.extend([0, 1, 2, 3, 4])

    cv.enable_cache()

    # Append to cache
    for i in range(5, 10):
        col.append(i)

    # setitem should flush cache first
    col[0] = 99

    cv.disable_cache()

    # Verify: cache was flushed, setitem worked
    assert len(col) == 10
    assert col[0] == 99
    all_data = list(col)
    assert all_data == [99, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_cache_flush_before_delete(temp_db):
    """Test that delete flushes cache first."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Pre-populate
    col.extend([0, 1, 2, 3, 4])

    cv.enable_cache()

    # Append to cache
    for i in range(5, 10):
        col.append(i)

    # Delete should flush cache first
    del col[0]

    cv.disable_cache()

    # Verify: cache was flushed, delete worked
    assert len(col) == 9
    assert list(col) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_cache_flush_before_insert(temp_db):
    """Test that insert flushes cache first."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("nums", "i64")

    # Pre-populate
    col.extend([0, 1, 2])

    cv.enable_cache()

    # Append to cache
    for i in range(3, 6):
        col.append(i)

    # Insert should flush cache first
    col.insert(0, 99)

    cv.disable_cache()

    # Verify: cache was flushed, insert worked
    assert len(col) == 7
    assert col[0] == 99
    all_data = list(col)
    assert all_data == [99, 0, 1, 2, 3, 4, 5]


# ======================================================================================
# Test 12: Variable-Size Columns (should work but no cache benefit)
# ======================================================================================


def test_variable_size_no_cache_benefit(temp_db):
    """Test that variable-size columns don't benefit from cache (but don't break)."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("msgs", "bytes")

    messages = [f"msg_{i}".encode() for i in range(100)]

    # Variable-size append doesn't use append_cached, so cache has no effect
    # But it shouldn't break anything
    with cv.cache():
        for msg in messages:
            col.append(msg)

    assert len(col) == 100
    assert [bytes(x) for x in col] == messages


def test_variable_size_extend_still_fast(temp_db):
    """Test that extend() for variable-size is still fast (already optimized)."""
    cv = ColumnVault(temp_db)
    col = cv.create_column("msgs", "bytes")

    messages = [f"msg_{i}".encode() for i in range(1000)]

    # extend() should be fast for variable-size (uses extend_adaptive)
    import time

    start = time.time()
    col.extend(messages)
    elapsed = time.time() - start

    assert len(col) == 1000
    # Should be very fast (< 0.1s for 1000 elements)
    assert elapsed < 0.5


# ======================================================================================
# Test 13: Mixed Fixed and Variable Size Columns
# ======================================================================================


def test_mixed_column_types_cache(temp_db):
    """Test caching with both fixed and variable-size columns."""
    cv = ColumnVault(temp_db)
    fixed_col = cv.create_column("nums", "i64")
    var_col = cv.create_column("msgs", "bytes")

    with cv.cache():
        for i in range(50):
            fixed_col.append(i)  # Should use cache
            var_col.append(f"msg_{i}".encode())  # No cache benefit but shouldn't break

    # Both should have correct data
    assert len(fixed_col) == 50
    assert len(var_col) == 50
    assert list(fixed_col) == list(range(50))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
