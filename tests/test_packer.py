"""
Unit tests for DataPacker.

Tests Rust-based data packing/unpacking for columnar storage.
"""

import pytest
from kohakuvault._kvault import DataPacker


class TestDataPackerBasics:
    """Test basic DataPacker construction and properties."""

    def test_i64_construction(self):
        packer = DataPacker("i64")
        assert packer.elem_size == 8
        assert not packer.is_varsize

    def test_f64_construction(self):
        packer = DataPacker("f64")
        assert packer.elem_size == 8
        assert not packer.is_varsize

    def test_string_varsize(self):
        packer = DataPacker("str:utf8")
        assert packer.elem_size == 0
        assert packer.is_varsize

    def test_string_fixed_size(self):
        packer = DataPacker("str:32:utf8")
        assert packer.elem_size == 32
        assert not packer.is_varsize

    def test_bytes_varsize(self):
        packer = DataPacker("bytes")
        assert packer.elem_size == 0
        assert packer.is_varsize

    def test_bytes_fixed_size(self):
        packer = DataPacker("bytes:128")
        assert packer.elem_size == 128
        assert not packer.is_varsize

    def test_msgpack(self):
        packer = DataPacker("msgpack")
        assert packer.elem_size == 0
        assert packer.is_varsize

    def test_cbor(self):
        packer = DataPacker("cbor")
        assert packer.elem_size == 0
        assert packer.is_varsize

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            DataPacker("invalid_type")


class TestInteger64Packing:
    """Test i64 packing/unpacking."""

    def test_pack_positive(self):
        packer = DataPacker("i64")
        packed = packer.pack(42)
        assert len(packed) == 8
        assert isinstance(packed, bytes)

    def test_pack_negative(self):
        packer = DataPacker("i64")
        packed = packer.pack(-42)
        assert len(packed) == 8

    def test_pack_zero(self):
        packer = DataPacker("i64")
        packed = packer.pack(0)
        assert len(packed) == 8

    def test_pack_max_value(self):
        packer = DataPacker("i64")
        max_val = 2**63 - 1
        packed = packer.pack(max_val)
        assert len(packed) == 8

    def test_pack_min_value(self):
        packer = DataPacker("i64")
        min_val = -(2**63)
        packed = packer.pack(min_val)
        assert len(packed) == 8

    def test_unpack_roundtrip(self):
        packer = DataPacker("i64")
        original = 12345
        packed = packer.pack(original)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == original

    def test_unpack_negative_roundtrip(self):
        packer = DataPacker("i64")
        original = -67890
        packed = packer.pack(original)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == original

    def test_pack_many(self):
        packer = DataPacker("i64")
        values = [1, 2, 3, 4, 5]
        packed = packer.pack_many(values)
        assert len(packed) == 40  # 5 * 8 bytes

    def test_unpack_many(self):
        packer = DataPacker("i64")
        values = [10, 20, 30, 40, 50]
        packed = packer.pack_many(values)
        unpacked = packer.unpack_many(packed, count=5)
        assert unpacked == values

    def test_unpack_with_offset(self):
        packer = DataPacker("i64")
        packed = packer.pack_many([100, 200, 300])

        assert packer.unpack(packed, 0) == 100
        assert packer.unpack(packed, 8) == 200
        assert packer.unpack(packed, 16) == 300


class TestFloat64Packing:
    """Test f64 packing/unpacking."""

    def test_pack_positive(self):
        packer = DataPacker("f64")
        packed = packer.pack(3.14159)
        assert len(packed) == 8

    def test_pack_negative(self):
        packer = DataPacker("f64")
        packed = packer.pack(-2.71828)
        assert len(packed) == 8

    def test_pack_zero(self):
        packer = DataPacker("f64")
        packed = packer.pack(0.0)
        assert len(packed) == 8

    def test_unpack_roundtrip(self):
        packer = DataPacker("f64")
        original = 3.14159265358979
        packed = packer.pack(original)
        unpacked = packer.unpack(packed, 0)
        assert abs(unpacked - original) < 1e-15

    def test_pack_many(self):
        packer = DataPacker("f64")
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        packed = packer.pack_many(values)
        assert len(packed) == 40  # 5 * 8 bytes

    def test_unpack_many(self):
        packer = DataPacker("f64")
        values = [1.5, 2.5, 3.5]
        packed = packer.pack_many(values)
        unpacked = packer.unpack_many(packed, count=3)

        for orig, unpacked_val in zip(values, unpacked):
            assert abs(orig - unpacked_val) < 1e-15


class TestStringPacking:
    """Test string packing with various encodings."""

    def test_utf8_variable_size(self):
        packer = DataPacker("str:utf8")
        text = "Hello, World!"
        packed = packer.pack(text)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_utf8_unicode(self):
        packer = DataPacker("str:utf8")
        text = "Hello, 世界! 🌍"
        packed = packer.pack(text)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_utf8_fixed_size(self):
        packer = DataPacker("str:32:utf8")
        text = "hello"
        packed = packer.pack(text)
        assert len(packed) == 32  # Padded to 32 bytes

        unpacked = packer.unpack(packed, 0)
        assert unpacked == text  # Padding should be trimmed

    def test_utf8_fixed_size_exact_fit(self):
        packer = DataPacker("str:5:utf8")
        text = "hello"
        packed = packer.pack(text)
        assert len(packed) == 5

        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_utf8_fixed_size_too_long(self):
        packer = DataPacker("str:5:utf8")
        text = "hello world"
        with pytest.raises(ValueError, match="Data too long"):
            packer.pack(text)

    def test_ascii_valid(self):
        packer = DataPacker("str:ascii")
        text = "Hello123"
        packed = packer.pack(text)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_ascii_invalid(self):
        packer = DataPacker("str:ascii")
        text = "Hello, 世界"
        with pytest.raises(ValueError, match="non-ASCII"):
            packer.pack(text)

    def test_latin1_valid(self):
        packer = DataPacker("str:latin1")
        text = "Café"
        packed = packer.pack(text)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_latin1_invalid(self):
        packer = DataPacker("str:latin1")
        text = "Hello, 世界"
        with pytest.raises(ValueError, match="outside Latin1 range"):
            packer.pack(text)

    def test_utf16le(self):
        packer = DataPacker("str:utf16le")
        text = "Hello"
        packed = packer.pack(text)
        # Each character = 2 bytes in UTF-16
        assert len(packed) == len(text) * 2

        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_utf16be(self):
        packer = DataPacker("str:utf16be")
        text = "Hello"
        packed = packer.pack(text)
        assert len(packed) == len(text) * 2

        unpacked = packer.unpack(packed, 0)
        assert unpacked == text

    def test_short_syntax(self):
        # "str:32" should default to UTF-8
        packer = DataPacker("str:32")
        text = "test"
        packed = packer.pack(text)
        assert len(packed) == 32


class TestBytesPacking:
    """Test raw bytes packing."""

    def test_variable_size(self):
        packer = DataPacker("bytes")
        data = b"\x00\x01\x02\x03\xff"
        packed = packer.pack(data)
        assert packed == data

        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_fixed_size(self):
        packer = DataPacker("bytes:128")
        data = b"test data"
        packed = packer.pack(data)
        assert len(packed) == 128  # Padded to 128 bytes

        unpacked = packer.unpack(packed, 0)
        # Unpacked should preserve padding
        assert len(unpacked) == 128
        assert unpacked[: len(data)] == data

    def test_fixed_size_exact_fit(self):
        packer = DataPacker("bytes:10")
        data = b"1234567890"
        packed = packer.pack(data)
        assert len(packed) == 10

        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_fixed_size_too_long(self):
        packer = DataPacker("bytes:5")
        data = b"1234567890"
        with pytest.raises(ValueError, match="Data too long"):
            packer.pack(data)

    def test_empty_bytes(self):
        packer = DataPacker("bytes")
        data = b""
        packed = packer.pack(data)
        assert packed == data

        unpacked = packer.unpack(packed, 0)
        assert unpacked == data


class TestMessagePackPacking:
    """Test MessagePack structured data packing."""

    def test_simple_dict(self):
        packer = DataPacker("msgpack")
        data = {"key": "value", "number": 42}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_nested_dict(self):
        packer = DataPacker("msgpack")
        data = {"user": {"name": "Alice", "age": 30, "profile": {"bio": "Software engineer"}}}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_list(self):
        packer = DataPacker("msgpack")
        data = [1, 2, 3, "four", 5.5, True, None]
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_mixed_types(self):
        packer = DataPacker("msgpack")
        data = {
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
        }
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_unicode_strings(self):
        packer = DataPacker("msgpack")
        data = {"message": "Hello, 世界! 🌍"}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_empty_dict(self):
        packer = DataPacker("msgpack")
        data = {}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_empty_list(self):
        packer = DataPacker("msgpack")
        data = []
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data


class TestCBORPacking:
    """Test CBOR structured data packing."""

    def test_simple_dict(self):
        packer = DataPacker("cbor")
        data = {"key": "value", "number": 42}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data

    def test_nested_structure(self):
        packer = DataPacker("cbor")
        data = {"user": {"name": "Bob", "scores": [95, 87, 92]}}
        packed = packer.pack(data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == data


@pytest.mark.skipif(
    not hasattr(DataPacker, "with_json_schema"), reason="schema-validation feature not enabled"
)
class TestJSONSchemaValidation:
    """Test JSON Schema validation for MessagePack."""

    def test_valid_data(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
            "required": ["name", "age"],
        }

        packer = DataPacker.with_json_schema(schema)

        # Valid data should work
        valid_data = {"name": "Alice", "age": 30}
        packed = packer.pack(valid_data)
        unpacked = packer.unpack(packed, 0)
        assert unpacked == valid_data

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        packer = DataPacker.with_json_schema(schema)

        # Missing required field should fail
        with pytest.raises(ValueError, match="Validation errors"):
            packer.pack({"name": "Bob"})

    def test_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        packer = DataPacker.with_json_schema(schema)

        # Wrong type should fail
        with pytest.raises(ValueError):
            packer.pack({"name": "Charlie", "age": "thirty"})

    def test_range_validation(self):
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 150}},
        }

        packer = DataPacker.with_json_schema(schema)

        # Valid range
        packer.pack({"age": 30})

        # Out of range
        with pytest.raises(ValueError):
            packer.pack({"age": -5})

        with pytest.raises(ValueError):
            packer.pack({"age": 200})


class TestPackerErrors:
    """Test error handling."""

    def test_unpack_not_enough_data(self):
        packer = DataPacker("i64")
        short_data = b"\x00\x01\x02"  # Only 3 bytes, need 8

        with pytest.raises(ValueError, match="Not enough data"):
            packer.unpack(short_data, 0)

    def test_unpack_many_variable_size_with_offsets(self):
        # Variable-size types can use unpack_many with offsets
        packer = DataPacker("str:utf8")
        strings = ["hello", "world", "test"]
        packed = packer.pack_many(strings)

        # Calculate offsets
        offsets = []
        pos = 0
        for s in strings:
            offsets.append(pos)
            pos += len(s.encode("utf-8"))

        # Unpack with offsets
        unpacked = packer.unpack_many(packed, offsets=offsets)
        assert unpacked == strings

    def test_unpack_many_msgpack_with_offsets(self):
        # MessagePack with offsets
        packer = DataPacker("msgpack")
        records = [{"a": 1}, {"b": 2}, {"c": 3}]

        # Pack each individually to know sizes
        packed_items = [packer.pack(r) for r in records]
        packed_all = b"".join(packed_items)

        # Calculate offsets
        offsets = []
        pos = 0
        for item in packed_items:
            offsets.append(pos)
            pos += len(item)

        # Unpack with offsets
        unpacked = packer.unpack_many(packed_all, offsets=offsets)
        assert unpacked == records

    def test_unpack_many_variable_size_without_offsets(self):
        # Variable-size without offsets should raise error
        packer = DataPacker("msgpack")

        with pytest.raises(ValueError, match="require offsets"):
            packer.unpack_many(b"", count=5)

    def test_pack_many_variable_size(self):
        # pack_many should work for variable-size, it concatenates packed bytes
        packer = DataPacker("msgpack")
        values = [{"a": 1}, {"b": 2}]
        packed = packer.pack_many(values)
        # Should not raise, each dict is packed separately
        assert len(packed) > 0

    def test_pack_many_strings(self):
        # pack_many works for variable strings
        packer = DataPacker("str:utf8")
        strings = ["hello", "world", "test"]
        packed = packer.pack_many(strings)

        # Each string is packed separately (UTF-8 encoded)
        expected_size = sum(len(s.encode("utf-8")) for s in strings)
        assert len(packed) == expected_size

    def test_pack_many_msgpack_list_of_dicts(self):
        # pack_many with list of dicts (common use case)
        packer = DataPacker("msgpack")
        records = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.3},
            {"id": 3, "name": "Charlie", "score": 92.1},
        ]
        packed = packer.pack_many(records)

        # Verify it packed successfully
        assert len(packed) > 0
        # Each record should be separately packed MessagePack bytes
        # Cannot unpack_many (no way to know boundaries), but pack_many works


class TestPackerIntegration:
    """Test DataPacker with Column integration."""

    def test_column_uses_rust_packer(self):
        """Verify that Column uses Rust packer by default."""
        from kohakuvault import ColumnVault

        vault = ColumnVault(":memory:")
        col = vault.create_column("test", "i64")

        # Should use Rust packer (verify by checking it works)
        col.append(42)
        col.extend([1, 2, 3, 4, 5])

        assert len(col) == 6
        assert col[0] == 42
        assert list(col) == [42, 1, 2, 3, 4, 5]

    def test_column_python_packer_fallback(self):
        """Test that Python packer fallback works."""
        from kohakuvault import ColumnVault

        vault = ColumnVault(":memory:")
        col = vault.create_column("test", "i64", use_rust_packer=False)

        col.append(100)
        col.extend([200, 300])

        assert len(col) == 3
        assert col[0] == 100
        assert col[1] == 200
        assert col[2] == 300
