"""Tests for auto-packing functionality."""

import numpy as np

from kohakuvault import Cbor, Json, KVault, MsgPack, Pickle


def test_auto_pack_enabled_by_default():
    """Test that auto-pack is enabled by default."""
    kv = KVault(":memory:")
    assert kv.auto_pack_enabled() is True
    kv.close()


def test_auto_pack_numpy_array():
    """Test auto-packing numpy arrays."""
    kv = KVault(":memory:")

    # Put numpy array
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    kv["embedding"] = vec

    # Get returns numpy array, not bytes!
    retrieved = kv["embedding"]
    assert isinstance(retrieved, np.ndarray)
    assert retrieved.dtype == np.float32
    np.testing.assert_array_almost_equal(retrieved, vec)

    kv.close()


def test_auto_pack_dict():
    """Test auto-packing dicts to MessagePack."""
    kv = KVault(":memory:")

    # Put dict
    config = {"timeout": 30, "retries": 3, "enabled": True}
    kv["config"] = config

    # Get returns dict, not bytes!
    retrieved = kv["config"]
    assert isinstance(retrieved, dict)
    assert retrieved == config

    kv.close()


def test_auto_pack_list():
    """Test auto-packing lists to MessagePack."""
    kv = KVault(":memory:")

    # Put list
    items = [1, 2, 3, 4, 5]
    kv["items"] = items

    # Get returns list, not bytes!
    retrieved = kv["items"]
    assert isinstance(retrieved, list)
    assert retrieved == items

    kv.close()


def test_auto_pack_int():
    """Test auto-packing integers."""
    kv = KVault(":memory:")

    kv["count"] = 42
    retrieved = kv["count"]
    assert isinstance(retrieved, int)
    assert retrieved == 42

    kv.close()


def test_auto_pack_float():
    """Test auto-packing floats."""
    kv = KVault(":memory:")

    kv["score"] = 95.5
    retrieved = kv["score"]
    assert isinstance(retrieved, float)
    assert abs(retrieved - 95.5) < 1e-10

    kv.close()


def test_auto_pack_string():
    """Test auto-packing strings."""
    kv = KVault(":memory:")

    kv["name"] = "Alice"
    retrieved = kv["name"]
    assert isinstance(retrieved, str)
    assert retrieved == "Alice"

    kv.close()


def test_auto_pack_bytes_stays_raw():
    """Test that bytes stay raw (no header)."""
    kv = KVault(":memory:")

    # Put bytes
    data = b"\xff\xd8\xff\xe0" + b"jpeg data"
    kv["image.jpg"] = data

    # Get returns bytes (raw, no encoding)
    retrieved = kv["image.jpg"]
    assert isinstance(retrieved, bytes)
    assert retrieved == data

    # Should NOT have header magic
    assert not retrieved.startswith(b"\x89K")

    kv.close()


def test_wrapped_msgpack():
    """Test explicit MsgPack wrapper."""
    kv = KVault(":memory:")

    data = {"key": "value"}
    kv["wrapped"] = MsgPack(data)

    retrieved = kv["wrapped"]
    assert isinstance(retrieved, dict)
    assert retrieved == data

    kv.close()


def test_wrapped_json():
    """Test explicit Json wrapper."""
    kv = KVault(":memory:")

    data = {"key": "value"}
    kv["json_data"] = Json(data)

    retrieved = kv["json_data"]
    assert isinstance(retrieved, dict)
    assert retrieved == data

    kv.close()


def test_auto_pack_nested_structure():
    """Test auto-packing complex nested structures."""
    kv = KVault(":memory:")

    data = {
        "users": [
            {"id": 1, "name": "Alice", "scores": [95, 87, 92]},
            {"id": 2, "name": "Bob", "scores": [88, 90, 85]},
        ],
        "metadata": {"created": "2025-11-08", "version": 1},
    }

    kv["complex"] = data
    retrieved = kv["complex"]

    assert isinstance(retrieved, dict)
    assert retrieved == data
    assert retrieved["users"][0]["name"] == "Alice"

    kv.close()


def test_disable_enable_auto_pack():
    """Test disabling and re-enabling auto-pack."""
    kv = KVault(":memory:")

    assert kv.auto_pack_enabled() is True

    # With auto-pack, can store dicts
    kv["dict1"] = {"key": "value"}
    assert kv["dict1"] == {"key": "value"}

    # Disable auto-pack
    kv.disable_auto_pack()
    assert kv.auto_pack_enabled() is False

    # Now must use bytes
    kv["bytes_key"] = b"raw bytes"
    assert kv["bytes_key"] == b"raw bytes"

    # Re-enable
    kv.enable_auto_pack()
    assert kv.auto_pack_enabled() is True

    kv["dict2"] = {"another": "dict"}
    assert kv["dict2"] == {"another": "dict"}

    kv.close()


def test_mixed_types_same_vault():
    """Test storing different types in same vault."""
    kv = KVault(":memory:")

    # Store different types
    kv["numpy"] = np.random.randn(10).astype(np.float32)
    kv["dict"] = {"key": "value"}
    kv["list"] = [1, 2, 3]
    kv["int"] = 42
    kv["float"] = 3.14
    kv["str"] = "hello"
    kv["bytes"] = b"raw data"

    # All retrieved correctly
    assert isinstance(kv["numpy"], np.ndarray)
    assert isinstance(kv["dict"], dict)
    assert isinstance(kv["list"], list)
    assert isinstance(kv["int"], int)
    assert isinstance(kv["float"], float)
    assert isinstance(kv["str"], str)
    assert isinstance(kv["bytes"], bytes)

    kv.close()


def test_backward_compat_old_values():
    """Test that old values (no auto-pack) still work."""
    kv = KVault(":memory:")

    # Disable auto-pack temporarily to write old-style data
    kv.disable_auto_pack()
    kv["old_key"] = b"old bytes value"

    # Re-enable auto-pack
    kv.enable_auto_pack()

    # Old value still readable (as bytes)
    old_value = kv["old_key"]
    assert old_value == b"old bytes value"

    # New values auto-packed
    kv["new_key"] = {"new": "data"}
    new_value = kv["new_key"]
    assert new_value == {"new": "data"}

    kv.close()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
