"""Basic tests for KohakuVault."""

import pytest
from kohakuvault import KVault, NotFound


def test_import():
    """Test that the module can be imported."""
    assert KVault is not None


def test_basic_operations():
    """Test basic put/get/delete operations."""
    vault = KVault(":memory:")

    # Put
    vault["key1"] = b"value1"
    assert vault["key1"] == b"value1"

    # Get
    assert vault.get("key1") == b"value1"
    assert vault.get("missing", b"default") == b"default"

    # Exists
    assert "key1" in vault
    assert "missing" not in vault

    # Delete
    del vault["key1"]
    assert "key1" not in vault

    vault.close()


def test_dict_interface():
    """Test dict-like interface."""
    vault = KVault(":memory:")

    # Set multiple keys
    vault["a"] = b"1"
    vault["b"] = b"2"
    vault["c"] = b"3"

    # Length
    assert len(vault) == 3

    # Iteration
    keys = list(vault.keys())
    assert len(keys) == 3
    assert b"a" in keys or "a" in keys  # Could be bytes or str

    # Clear
    vault.clear()
    assert len(vault) == 0

    vault.close()


def test_not_found():
    """Test KeyError for missing keys (dict-like behavior)."""
    vault = KVault(":memory:")

    # Dict-like interface raises KeyError (standard Python behavior)
    with pytest.raises(KeyError):
        _ = vault["missing_key"]

    # get() with default returns default (doesn't raise)
    assert vault.get("missing_key", b"default") == b"default"

    # get() without default returns None
    assert vault.get("missing_key") is None

    vault.close()


def test_context_manager():
    """Test context manager protocol."""
    with KVault(":memory:") as vault:
        vault["test"] = b"data"
        assert vault["test"] == b"data"
    # Vault should be closed after context


def test_cache():
    """Test write-back cache."""
    vault = KVault(":memory:")

    # Enable cache
    vault.enable_cache(cap_bytes=1024 * 1024, flush_threshold=512 * 1024)

    # Write data
    for i in range(10):
        vault[f"key{i}"] = b"value"

    # Flush cache
    count = vault.flush_cache()
    assert count == 10

    # Verify data persisted
    assert vault["key0"] == b"value"

    vault.disable_cache()
    vault.close()


def test_binary_keys():
    """Test binary keys work correctly."""
    vault = KVault(":memory:")

    key = b"\x00\x01\x02\xff"
    value = b"binary data"

    vault[key] = value
    assert vault[key] == value

    vault.close()


def test_delete_operations():
    """Test various delete operations."""
    vault = KVault(":memory:")

    # Setup
    vault["key1"] = b"value1"

    # Delete existing key
    del vault["key1"]
    assert "key1" not in vault

    # Delete non-existent key raises KeyError
    with pytest.raises(KeyError):
        del vault["missing"]

    # delete() method returns True for existing key
    vault["key2"] = b"value2"
    assert vault.delete("key2") is True

    # delete() returns False for non-existent key (doesn't raise)
    assert vault.delete("missing") is False

    vault.close()
