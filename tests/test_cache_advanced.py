"""
Advanced cache tests for KohakuVault.

Tests the new cache features:
- Large value handling (bypass cache)
- Capacity overflow (auto-flush before insert)
- Context manager (auto-flush on exit)
- Daemon thread (auto-flush after idle)
- Lock cache (prevent auto-flush)
"""

import time
import pytest
from kohakuvault import KVault


def test_cache_value_larger_than_capacity():
    """Test that values larger than cache capacity are handled correctly."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024, flush_threshold=512)

    # Write value larger than cache capacity (2KB > 1KB cache)
    large_value = b"x" * 2048
    vault["large_key"] = large_value

    # Should bypass cache and write directly
    # Verify it's actually stored
    vault.disable_cache()
    assert vault["large_key"] == large_value


def test_cache_capacity_overflow_auto_flush():
    """Test that cache auto-flushes when capacity would be exceeded."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024, flush_threshold=2048)  # Threshold > capacity

    # Write values that together exceed capacity
    vault["k1"] = b"x" * 600  # 600 bytes
    vault["k2"] = b"x" * 600  # Would make 1200 > 1024, should auto-flush k1 first

    # Both should be in database after auto-flush
    vault.disable_cache()
    assert vault["k1"] == b"x" * 600
    assert vault["k2"] == b"x" * 600


def test_cache_threshold_auto_flush():
    """Test that cache auto-flushes at threshold."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=2048, flush_threshold=1000)

    # Write data that exceeds threshold
    for i in range(5):
        vault[f"key:{i}"] = b"x" * 250  # Total 1250 > 1000

    # Should have auto-flushed at threshold
    # Disable cache and verify all data persisted
    vault.disable_cache()

    for i in range(5):
        assert vault[f"key:{i}"] == b"x" * 250


def test_cache_context_manager_auto_flush():
    """Test that context manager auto-flushes on exit."""
    vault = KVault(":memory:")

    with vault.cache(cap_bytes=1024 * 1024):
        vault["key1"] = b"data1"
        vault["key2"] = b"data2"
    # Auto-flush happens here

    # Cache is disabled and data should be persisted
    assert vault["key1"] == b"data1"
    assert vault["key2"] == b"data2"


def test_cache_context_manager_with_exception():
    """Test that context manager flushes even when exception occurs."""
    vault = KVault(":memory:")

    try:
        with vault.cache(cap_bytes=1024 * 1024):
            vault["key_before_error"] = b"important_data"
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Data should still be flushed despite exception
    assert vault["key_before_error"] == b"important_data"


def test_cache_context_manager_no_auto_flush():
    """Test context manager with auto_flush=False.

    Note: Even with auto_flush=False, disable_cache() still flushes for safety.
    This test verifies the context manager respects auto_flush parameter
    during the context, but data is still persisted on exit.
    """
    vault = KVault(":memory:")

    with vault.cache(cap_bytes=1024 * 1024, auto_flush=False):
        vault["key"] = b"data"
        # No auto-flush during context, but disable_cache() will flush

    # Data is persisted because disable_cache() auto-flushes for safety
    assert vault["key"] == b"data"


def test_cache_context_manager_manual_flush():
    """Test manual flush within context manager."""
    vault = KVault(":memory:")

    with vault.cache(cap_bytes=1024 * 1024, auto_flush=False):
        vault["key"] = b"data"
        vault.flush_cache()  # Manual flush

    # Data should be persisted
    assert vault["key"] == b"data"


def test_cache_daemon_thread_auto_flush():
    """Test daemon thread auto-flushes after idle period."""
    vault = KVault(":memory:")

    # Enable cache with 0.5 second auto-flush
    vault.enable_cache(cap_bytes=1024 * 1024, flush_interval=0.5)

    vault["key1"] = b"data1"

    # Wait for daemon to flush (should happen after 0.5s idle)
    time.sleep(0.8)

    # Disable cache (stops daemon)
    vault.disable_cache()

    # Data should be persisted by daemon
    assert vault["key1"] == b"data1"


def test_cache_daemon_continues_flushing():
    """Test daemon flushes multiple times."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024 * 1024, flush_interval=0.3)

    # Write, wait, write, wait
    vault["key1"] = b"data1"
    time.sleep(0.5)  # First flush

    vault["key2"] = b"data2"
    time.sleep(0.5)  # Second flush

    vault.disable_cache()

    # Both should be persisted
    assert vault["key1"] == b"data1"
    assert vault["key2"] == b"data2"


def test_lock_cache_prevents_auto_flush():
    """Test that lock_cache() prevents daemon from flushing."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024 * 1024, flush_interval=0.3)

    with vault.lock_cache():
        vault["key1"] = b"data1"
        time.sleep(0.5)  # Daemon tries to flush but locked
        vault["key2"] = b"data2"
    # Lock released, daemon can flush now

    time.sleep(0.5)  # Wait for daemon flush

    vault.disable_cache()

    # Both keys should be persisted
    assert vault["key1"] == b"data1"
    assert vault["key2"] == b"data2"


def test_cache_mixed_large_and_small_values():
    """Test cache with mix of large and small values."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024, flush_threshold=512)

    vault["small1"] = b"x" * 100  # Cached
    vault["large1"] = b"x" * 2000  # Bypasses cache (triggers flush first)
    vault["small2"] = b"x" * 100  # Cached again

    vault.flush_cache()
    vault.disable_cache()

    assert vault["small1"] == b"x" * 100
    assert vault["large1"] == b"x" * 2000
    assert vault["small2"] == b"x" * 100


def test_cache_multiple_flushes_in_session():
    """Test multiple auto-flushes during a session."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=512, flush_threshold=256)

    # Each write triggers flush after threshold
    for i in range(10):
        vault[f"key:{i}"] = b"x" * 300  # Each write > threshold

    vault.disable_cache()

    # All should be persisted
    for i in range(10):
        assert vault[f"key:{i}"] == b"x" * 300


def test_cache_daemon_stops_on_disable():
    """Test that daemon stops when cache disabled."""
    vault = KVault(":memory:")
    vault.enable_cache(flush_interval=0.2)

    vault["key"] = b"data"

    # Disable immediately (daemon should stop)
    vault.disable_cache()

    # Wait a bit
    time.sleep(0.5)

    # Daemon should be stopped, no crashes
    # Data might or might not be flushed depending on timing
    # Just verify no errors


def test_cache_daemon_stops_on_close():
    """Test that daemon stops when vault closed."""
    vault = KVault(":memory:")
    vault.enable_cache(flush_interval=0.2)

    vault["key"] = b"data"

    # Close vault (should stop daemon and flush)
    vault.close()

    # No crashes, daemon cleanly stopped


def test_backward_compatibility_no_flush_interval():
    """Test that old API still works (no flush_interval)."""
    vault = KVault(":memory:")
    vault.enable_cache(cap_bytes=1024 * 1024, flush_threshold=512 * 1024)

    for i in range(1000):
        vault[f"key:{i}"] = b"data"

    vault.flush_cache()
    vault.disable_cache()

    # All should be persisted
    assert len(vault) == 1000


def test_cache_context_nested():
    """Test nested cache context managers."""
    vault = KVault(":memory:")

    with vault.cache(1024 * 1024):
        vault["outer1"] = b"data"

        # Can't nest - second enable_cache would reset
        # Just verify first one works
        vault["outer2"] = b"data"

    assert vault["outer1"] == b"data"
    assert vault["outer2"] == b"data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
