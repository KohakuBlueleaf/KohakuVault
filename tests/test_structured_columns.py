"""
Tests for structured dtype columns (msgpack, cbor, strings).

Tests the ability to store structured data (dicts, lists) and strings
in columnar storage using DataPacker.
"""

import pytest
from kohakuvault import ColumnVault


class TestMessagePackColumns:
    """Test MessagePack columns for structured data storage."""

    def test_msgpack_simple_dicts(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("users", "msgpack")

        # Append dicts
        col.append({"name": "Alice", "age": 30})
        col.append({"name": "Bob", "age": 25})
        col.append({"name": "Charlie", "age": 35})

        assert len(col) == 3
        assert col[0] == {"name": "Alice", "age": 30}
        assert col[1] == {"name": "Bob", "age": 25}
        assert col[2] == {"name": "Charlie", "age": 35}

    def test_msgpack_nested_structures(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("data", "msgpack")

        # Complex nested structure
        data = {
            "user": {
                "id": 123,
                "profile": {"name": "Alice", "verified": True},
            },
            "tags": ["vip", "premium"],
            "metadata": {"created": "2025-01-01"},
        }

        col.append(data)
        retrieved = col[0]

        assert retrieved == data
        assert retrieved["user"]["profile"]["verified"] is True

    def test_msgpack_extend_bulk(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("records", "msgpack")

        # Bulk append
        records = [{"id": i, "value": f"item_{i}", "score": i * 1.5} for i in range(100)]

        col.extend(records)

        assert len(col) == 100
        assert col[0] == {"id": 0, "value": "item_0", "score": 0.0}
        assert col[99] == {"id": 99, "value": "item_99", "score": 148.5}

    def test_msgpack_mixed_types(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("mixed", "msgpack")

        # Different structures in same column
        col.append({"type": "user", "name": "Alice"})
        col.append({"type": "event", "action": "login", "timestamp": 1234567890})
        col.append(["list", "of", "strings"])
        col.append(42)  # Even primitive types work
        col.append(None)
        col.append(True)

        assert len(col) == 6
        assert col[0]["type"] == "user"
        assert col[1]["action"] == "login"
        assert col[2] == ["list", "of", "strings"]
        assert col[3] == 42
        assert col[4] is None
        assert col[5] is True

    def test_msgpack_iteration(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("items", "msgpack")

        items = [{"id": i, "name": f"item{i}"} for i in range(10)]
        col.extend(items)

        # Iteration should work
        retrieved = list(col)
        assert retrieved == items

    def test_msgpack_clear(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("data", "msgpack")

        col.extend([{"a": 1}, {"b": 2}, {"c": 3}])
        assert len(col) == 3

        col.clear()
        assert len(col) == 0


class TestCBORColumns:
    """Test CBOR columns."""

    def test_cbor_basic(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("cbor_data", "cbor")

        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        col.append(data)

        retrieved = col[0]
        assert retrieved == data

    def test_cbor_extend(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("cbor_records", "cbor")

        records = [{"id": i, "data": f"record_{i}"} for i in range(50)]
        col.extend(records)

        assert len(col) == 50
        assert col[25] == {"id": 25, "data": "record_25"}


class TestStringColumns:
    """Test variable-size string columns."""

    def test_utf8_strings(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("messages", "str:utf8")

        col.append("Hello, World!")
        col.append("Unicode: 世界 🌍")
        col.append("Short")
        col.append("A much longer string with lots of text...")

        assert len(col) == 4
        assert col[0] == "Hello, World!"
        assert col[1] == "Unicode: 世界 🌍"
        assert col[2] == "Short"
        assert col[3] == "A much longer string with lots of text..."

    def test_utf8_bulk_extend(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("text", "str:utf8")

        texts = [f"String number {i}" for i in range(100)]
        col.extend(texts)

        assert len(col) == 100
        assert col[50] == "String number 50"

    def test_ascii_strings(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("ascii_data", "str:ascii")

        col.append("Hello")
        col.append("World")
        col.append("123")

        assert len(col) == 3
        assert list(col) == ["Hello", "World", "123"]

    def test_ascii_validation_error(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("ascii_only", "str:ascii")

        # Non-ASCII should fail
        with pytest.raises(ValueError, match="non-ASCII"):
            col.append("Hello, 世界")

    def test_utf16le_strings(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("utf16_data", "str:utf16le")

        col.append("Hello")
        col.append("Windows")

        assert len(col) == 2
        assert col[0] == "Hello"
        assert col[1] == "Windows"

    def test_latin1_strings(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("latin1_data", "str:latin1")

        col.append("Café")
        col.append("Résumé")

        assert len(col) == 2
        assert col[0] == "Café"
        assert col[1] == "Résumé"


class TestMixedDataTypes:
    """Test mixing different dtypes in same vault."""

    def test_multiple_column_types(self):
        vault = ColumnVault(":memory:")

        # Create columns of different types
        ids = vault.create_column("ids", "i64")
        scores = vault.create_column("scores", "f64")
        names = vault.create_column("names", "str:utf8")
        metadata = vault.create_column("metadata", "msgpack")

        # Add data
        for i in range(10):
            ids.append(i)
            scores.append(i * 1.5)
            names.append(f"User_{i}")
            metadata.append({"created": "2025-01-01", "active": True})

        assert len(ids) == 10
        assert len(scores) == 10
        assert len(names) == 10
        assert len(metadata) == 10

        # Verify data integrity
        assert ids[5] == 5
        assert scores[5] == 7.5
        assert names[5] == "User_5"
        assert metadata[5]["active"] is True


class TestStructuredPersistence:
    """Test that structured data persists correctly."""

    def test_msgpack_persistence(self, tmp_path):
        db_path = tmp_path / "test_msgpack.db"

        # Write data
        vault1 = ColumnVault(str(db_path))
        col1 = vault1.create_column("data", "msgpack")
        test_data = [
            {"name": "Alice", "scores": [95, 87, 92]},
            {"name": "Bob", "scores": [88, 91, 85]},
        ]
        col1.extend(test_data)

        # Re-open and verify
        vault2 = ColumnVault(str(db_path))
        col2 = vault2["data"]

        assert len(col2) == 2
        assert col2[0] == test_data[0]
        assert col2[1] == test_data[1]

    def test_string_persistence(self, tmp_path):
        db_path = tmp_path / "test_strings.db"

        # Write data
        vault1 = ColumnVault(str(db_path))
        col1 = vault1.create_column("messages", "str:utf8")
        messages = ["Hello", "World", "Unicode: 世界"]
        col1.extend(messages)

        # Re-open and verify
        vault2 = ColumnVault(str(db_path))
        col2 = vault2["messages"]

        assert len(col2) == 3
        assert list(col2) == messages


class TestStructuredEdgeCases:
    """Test edge cases for structured columns."""

    def test_empty_dict(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("data", "msgpack")

        col.append({})
        assert col[0] == {}

    def test_empty_list(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("data", "msgpack")

        col.append([])
        assert col[0] == []

    def test_empty_string(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("text", "str:utf8")

        col.append("")
        assert col[0] == ""

    def test_large_msgpack_object(self):
        vault = ColumnVault(":memory:")
        col = vault.create_column("large", "msgpack")

        # Large nested structure
        large_obj = {
            "users": [{"id": i, "name": f"user_{i}", "data": list(range(100))} for i in range(50)]
        }

        col.append(large_obj)
        retrieved = col[0]

        assert retrieved == large_obj
        assert len(retrieved["users"]) == 50


class TestValidatorBehavior:
    """Test that parse_dtype correctly validates using DataPacker."""

    def test_valid_dtypes(self):
        vault = ColumnVault(":memory:")

        # All these should work
        vault.create_column("a", "i64")
        vault.create_column("b", "f64")
        vault.create_column("c", "bytes:128")
        vault.create_column("d", "bytes")
        vault.create_column("e", "str:utf8")
        vault.create_column("f", "str:32")
        vault.create_column("g", "msgpack")
        vault.create_column("h", "cbor")

        # Verify they were created
        # Note: Variable-size columns create 2 underlying columns (_data and _idx)
        # So we have: a, b, c, d_data, d_idx, e_data, e_idx, f, g_data, g_idx, h_data, h_idx
        # That's 3 fixed + 5 varsize*2 = 3 + 10 = 13, but we count logical columns
        cols = vault.list_columns()
        # Check that at least all base columns exist
        col_names = [c[0] for c in cols]
        assert "a" in col_names
        assert "b" in col_names
        assert "f" in col_names  # Fixed-size string

    def test_invalid_dtype(self):
        vault = ColumnVault(":memory:")

        with pytest.raises(Exception):  # Should raise InvalidArgument
            vault.create_column("bad", "invalid_type")

    def test_fixed_vs_varsize_detection(self):
        vault = ColumnVault(":memory:")

        # Fixed-size
        col_fixed = vault.create_column("fixed", "i64")
        assert hasattr(col_fixed, "_elem_size")
        assert col_fixed._elem_size == 8

        # Variable-size
        col_var = vault.create_column("var", "msgpack")
        assert isinstance(col_var, type(vault.create_column("test_bytes", "bytes")))
