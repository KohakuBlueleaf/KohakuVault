"""Tests for KVault header system."""

import tempfile
from pathlib import Path

import pytest

from kohakuvault import KVault


def test_headers_enabled_by_default():
    """Test that headers are enabled by default for auto-packing."""
    kv = KVault(":memory:")

    assert kv.headers_enabled() is True

    # Raw bytes should be stored as-is (no header)
    kv["key1"] = b"raw data"
    value = kv["key1"]
    assert value == b"raw data"

    # Should NOT have header magic bytes
    assert not value.startswith(b"\x89K")

    kv.close()


def test_enable_headers():
    """Test enabling header format."""
    kv = KVault(":memory:")

    # Enable headers
    kv.enable_headers()
    assert kv.headers_enabled() is True

    kv.close()


def test_disable_headers():
    """Test disabling header format."""
    kv = KVault(":memory:")

    kv.enable_headers()
    assert kv.headers_enabled() is True

    kv.disable_headers()
    assert kv.headers_enabled() is False

    kv.close()


def test_raw_bytes_no_header_even_when_enabled():
    """Test that raw bytes stay raw even when headers are enabled."""
    kv = KVault(":memory:")

    # Enable headers
    kv.enable_headers()

    # Put raw bytes - should still be stored without header
    # This ensures media files can be previewed by external tools
    kv["image.jpg"] = b"\xff\xd8\xff\xe0"  # JPEG magic bytes

    value = kv["image.jpg"]
    assert value == b"\xff\xd8\xff\xe0"
    # Should NOT have KohakuVault header
    assert not value.startswith(b"\x89K")

    kv.close()


def test_mixed_format_same_vault():
    """Test that old format (no header) and new format can coexist."""
    kv = KVault(":memory:")

    # Write some values without headers (old format)
    kv["old1"] = b"old data 1"
    kv["old2"] = b"old data 2"

    # Enable headers
    kv.enable_headers()

    # Write new values (still raw bytes, so no headers added)
    kv["new1"] = b"new data 1"

    # All should be readable
    assert kv["old1"] == b"old data 1"
    assert kv["old2"] == b"old data 2"
    assert kv["new1"] == b"new data 1"

    # Disable headers and write more
    kv.disable_headers()
    kv["new2"] = b"new data 2"

    # All still readable
    assert kv["old1"] == b"old data 1"
    assert kv["new2"] == b"new data 2"

    kv.close()


def test_meta_table_created():
    """Test that meta table is created and accessible."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        kv = KVault(db_path)
        kv.close()
        del kv  # Explicitly delete to release file handles

        # Check that meta table exists
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()

            # Check table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kohakuvault_meta'"
            )
            result = cursor.fetchone()
            assert result is not None
        finally:
            cursor.close()
            conn.close()

    finally:
        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
        Path(f"{db_path}-wal").unlink(missing_ok=True)


def test_feature_registration():
    """Test that enabling headers registers the feature in meta table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        kv = KVault(db_path)

        # Enable headers
        kv.enable_headers()
        kv.close()
        del kv  # Explicitly delete to release file handles

        # Check meta table has the feature registered
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT value FROM kohakuvault_meta WHERE key='kv_features'")
            result = cursor.fetchone()

            assert result is not None
            features = result[0]
            assert "headers_v1" in features
        finally:
            cursor.close()
            conn.close()

    finally:
        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
        Path(f"{db_path}-wal").unlink(missing_ok=True)


def test_backward_compatibility_with_old_db():
    """Test that old databases without meta table still work."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Create old-style database (just the kv table, no meta)
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE kv (key BLOB PRIMARY KEY NOT NULL, value BLOB NOT NULL)")
            conn.execute("INSERT INTO kv (key, value) VALUES (?, ?)", (b"key1", b"value1"))
            conn.commit()
        finally:
            conn.close()

        # Open with new KVault (should create meta table automatically)
        kv = KVault(db_path)

        # Old data should still be accessible
        assert kv["key1"] == b"value1"

        # Can write new data
        kv["key2"] = b"value2"
        assert kv["key2"] == b"value2"

        kv.close()
        del kv  # Explicitly delete to release file handles

        # Verify meta table was created
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kohakuvault_meta'"
            )
            assert cursor.fetchone() is not None
        finally:
            cursor.close()
            conn.close()

    finally:
        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
        Path(f"{db_path}-wal").unlink(missing_ok=True)


def test_header_preserved_across_sessions():
    """Test that header mode setting is preserved across sessions."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Session 1: Headers enabled by default, write data
        kv1 = KVault(db_path)
        assert kv1.headers_enabled() is True
        kv1["key1"] = b"data1"
        kv1.close()
        del kv1  # Explicitly delete to release file handles

        # Session 2: Reopen (headers enabled by default)
        kv2 = KVault(db_path)
        assert kv2.headers_enabled() is True

        # But feature should be registered in meta table
        # User can check and re-enable if needed

        # Data should still be readable
        assert kv2["key1"] == b"data1"

        kv2.close()
        del kv2  # Explicitly delete to release file handles

    finally:
        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
        Path(f"{db_path}-wal").unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
