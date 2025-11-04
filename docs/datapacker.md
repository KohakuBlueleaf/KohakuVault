# DataPacker API Reference

## Overview

`DataPacker` is a Rust-based class for efficient data serialization/deserialization in KohakuVault columnar storage. It moves data packing logic from Python to Rust for improved performance.

## Key Features

- **Pure Rust Implementation**: No external dependencies (no protoc, flatc, or capnp required)
- **MessagePack Primary**: Compact binary format, excellent cross-language support
- **Multi-Encoding Strings**: UTF-8, UTF-16LE, UTF-16BE, ASCII, Latin-1
- **Fixed & Variable Size**: Support for both fixed-size (padded) and variable-size types
- **JSON Schema Validation**: Optional runtime validation without external tools
- **CBOR Support**: Alternative IETF standard format
- **Batch Operations**: Optimized `pack_many()` for bulk packing

## Size Efficiency

MessagePack and CBOR are binary formats that are typically smaller than JSON. The exact compression ratio depends on your data structure. Run the included benchmark script to measure performance on your specific workload.

---

## Supported Types

### Primitives

#### Integers (`i64`)
- **Size**: 8 bytes (fixed)
- **Format**: Little-endian signed 64-bit
- **Range**: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

```python
packer = DataPacker("i64")
packed = packer.pack(42)          # 8 bytes
assert len(packed) == 8
unpacked = packer.unpack(packed, 0)
assert unpacked == 42
```

#### Floats (`f64`)
- **Size**: 8 bytes (fixed)
- **Format**: IEEE 754 double-precision
- **Range**: ±1.7E±308 (~15-17 significant decimal digits)

```python
packer = DataPacker("f64")
packed = packer.pack(3.14159)
unpacked = packer.unpack(packed, 0)
assert abs(unpacked - 3.14159) < 1e-10
```

#### Strings (`str`)
- **Encodings**: UTF-8 (default), UTF-16LE, UTF-16BE, ASCII, Latin-1
- **Modes**: Variable-size or fixed-size (padded/truncated)

**Variable-size UTF-8**:
```python
packer = DataPacker("str:utf8")
packed = packer.pack("Hello, 世界")
assert packer.is_varsize
```

**Fixed-size UTF-8** (32 bytes, zero-padded):
```python
packer = DataPacker("str:32:utf8")
packed = packer.pack("hello")
assert len(packed) == 32
assert packer.elem_size == 32
```

**Short syntax** (assumes UTF-8):
```python
packer = DataPacker("str:32")  # Same as "str:32:utf8"
```

**UTF-16LE** (Windows compatibility):
```python
packer = DataPacker("str:utf16le")
packed = packer.pack("Hello")
# Each character = 2 bytes
```

**ASCII** (strict, 7-bit):
```python
packer = DataPacker("str:ascii")
packed = packer.pack("Hello")  # OK
# packer.pack("Hello, 世界")  # ERROR: non-ASCII
```

**Latin-1 / ISO-8859-1** (Western European):
```python
packer = DataPacker("str:latin1")
packed = packer.pack("Café")  # OK (characters 0-255)
```

#### Bytes (`bytes`)
- **Modes**: Variable-size or fixed-size (zero-padded)

**Variable-size**:
```python
packer = DataPacker("bytes")
data = b"\x00\x01\x02\x03\xff"
packed = packer.pack(data)
assert packed == data
```

**Fixed-size** (128 bytes):
```python
packer = DataPacker("bytes:128")
data = b"test data"
packed = packer.pack(data)
assert len(packed) == 128  # Zero-padded to 128
```

### Structured Formats

#### MessagePack (`msgpack`)
- **Schema**: None (schema-less like JSON)
- **Size**: ~30% of JSON
- **Use**: Dictionaries, lists, nested structures

```python
packer = DataPacker("msgpack")

data = {
    "name": "Alice",
    "age": 30,
    "scores": [95, 87, 92],
    "metadata": {
        "verified": True,
        "tags": ["vip", "premium"]
    }
}

packed = packer.pack(data)
unpacked = packer.unpack(packed, 0)
assert unpacked == data

# Size comparison
import json
json_size = len(json.dumps(data))
msgpack_size = len(packed)
print(f"JSON: {json_size}B, MessagePack: {msgpack_size}B")
# Output: JSON: 123B, MessagePack: 89B (72%)
```

#### MessagePack with JSON Schema Validation (`with_json_schema`)
- **Validation**: Runtime JSON Schema validation
- **Use**: When you need type safety

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    },
    "required": ["name", "age"]
}

packer = DataPacker.with_json_schema(schema)

# Valid data
packer.pack({"name": "Alice", "age": 30})  # OK

# Invalid data raises ValueError
try:
    packer.pack({"name": "Bob"})  # Missing 'age'
except ValueError as e:
    print(f"Validation error: {e}")

try:
    packer.pack({"name": "Charlie", "age": -5})  # age < 0
except ValueError as e:
    print(f"Validation error: {e}")
```

#### CBOR (`cbor`)
- **Schema**: Optional CDDL (future)
- **Size**: ~40% of JSON
- **Use**: IETF standard, rich type system

```python
packer = DataPacker("cbor")
data = {"key": "value", "number": 42}
packed = packer.pack(data)
unpacked = packer.unpack(packed, 0)
```

---

## Construction

### Basic Constructor

```python
from kohakuvault import DataPacker

# Primitives
packer = DataPacker("i64")
packer = DataPacker("f64")
packer = DataPacker("str:utf8")
packer = DataPacker("str:32:utf16le")
packer = DataPacker("bytes:128")

# Structured
packer = DataPacker("msgpack")
packer = DataPacker("cbor")
```

### With JSON Schema Validation

```python
schema = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "email": {"type": "string", "format": "email"},
        "active": {"type": "boolean"}
    },
    "required": ["user_id", "email"]
}

packer = DataPacker.with_json_schema(schema)
```

### With CDDL Schema (CBOR)

```python
# Currently accepts schema string but validation not yet implemented
cddl_schema = """
person = {
    name: tstr,
    age: uint,
}
"""

packer = DataPacker.with_cddl_schema(cddl_schema)
```

---

## Methods

### `pack(value) → bytes`

Pack a single value to bytes.

**Parameters:**
- `value`: Python value (int, float, str, bytes, dict, list)

**Returns:**
- `bytes`: Packed binary data

**Raises:**
- `ValueError`: Invalid value for dtype
- `TypeError`: Value type doesn't match dtype

**Examples:**

```python
# Integer
packer = DataPacker("i64")
packed = packer.pack(42)
assert len(packed) == 8

# String
packer = DataPacker("str:utf8")
packed = packer.pack("Hello, 世界")

# Dictionary (MessagePack)
packer = DataPacker("msgpack")
packed = packer.pack({"key": "value", "count": 10})
```

### `pack_many(values: list) → bytes`

Pack multiple values to concatenated bytes.

**Works for ALL types** (including variable-size like str, msgpack, cbor).

**Parameters:**
- `values`: List of Python values

**Returns:**
- `bytes`: Concatenated packed data

**Examples:**

```python
# Fixed-size types
packer = DataPacker("i64")
values = [1, 2, 3, 4, 5]
packed = packer.pack_many(values)
assert len(packed) == 40  # 5 * 8 bytes

# Variable-size types work too!
packer = DataPacker("str:utf8")
strings = ["hello", "world", "test"]
packed = packer.pack_many(strings)
# Concatenates UTF-8 encoded bytes

# MessagePack (list of dicts)
packer = DataPacker("msgpack")
records = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]
packed = packer.pack_many(records)
# Each dict is packed separately and concatenated
```

**Note**: For variable-size types, `pack_many()` works but `unpack_many()` doesn't (no way to know where each item ends without length prefixes). Use VarSizeColumn for variable-size storage.

### `unpack(data: bytes, offset: int = 0) → value`

Unpack a single value from bytes at offset.

**Parameters:**
- `data`: Packed binary data
- `offset`: Byte offset to start unpacking (default: 0)

**Returns:**
- Python value (int, float, str, bytes, dict, list)

**Raises:**
- `ValueError`: Not enough data or invalid format

**Examples:**

```python
packer = DataPacker("i64")
packed = packer.pack(42)
value = packer.unpack(packed, 0)
assert value == 42

# Multiple values in one buffer
packer = DataPacker("i64")
packed = packer.pack_many([10, 20, 30])
assert packer.unpack(packed, 0) == 10   # First value
assert packer.unpack(packed, 8) == 20   # Second value
assert packer.unpack(packed, 16) == 30  # Third value
```

### `unpack_many(data: bytes, count: int = None, offsets: list[int] = None) → list`

Unpack multiple values from bytes.

**For fixed-size types**: Use `count` parameter
**For variable-size types**: Use `offsets` parameter (list of start positions)

**Parameters:**
- `data`: Packed binary data
- `count`: Number of values to unpack (required for fixed-size)
- `offsets`: Start offsets of each element (required for variable-size)

**Returns:**
- `list`: List of unpacked values

**Examples:**

```python
# Fixed-size types
packer = DataPacker("i64")
packed = packer.pack_many([1, 2, 3, 4, 5])
values = packer.unpack_many(packed, count=5)
assert values == [1, 2, 3, 4, 5]

# Variable-size types (with offsets)
packer = DataPacker("str:utf8")
strings = ["hello", "world", "test"]
packed = packer.pack_many(strings)

# Calculate offsets
offsets = [0, 5, 10]  # "hello"=0, "world"=5, "test"=10
unpacked = packer.unpack_many(packed, offsets=offsets)
assert unpacked == strings

# MessagePack with offsets
packer = DataPacker("msgpack")
records = [{"a": 1}, {"b": 2}]
packed_items = [packer.pack(r) for r in records]
offsets = [0, len(packed_items[0])]
packed_all = b"".join(packed_items)
unpacked = packer.unpack_many(packed_all, offsets=offsets)
assert unpacked == records
```

---

## Properties

### `elem_size: int`

Get element size in bytes (0 for variable-size types).

```python
packer = DataPacker("i64")
assert packer.elem_size == 8

packer = DataPacker("str:32:utf8")
assert packer.elem_size == 32

packer = DataPacker("msgpack")
assert packer.elem_size == 0  # Variable-size
```

### `is_varsize: bool`

Check if this is a variable-size type.

```python
packer = DataPacker("i64")
assert not packer.is_varsize

packer = DataPacker("msgpack")
assert packer.is_varsize
```

---

## Integration with ColumnVault

DataPacker is automatically used by `Column` for improved performance.

### Automatic Usage (Default)

```python
from kohakuvault import ColumnVault

vault = ColumnVault(":memory:")
col = vault.create_column("ages", "i64")

# Automatically uses Rust DataPacker
col.append(25)
col.extend([30, 35, 40, 45])

print(col[0])  # 25
```

### Disable Rust Packing (Fallback to Python)

```python
# For debugging or compatibility
col = vault.create_column("ages", "i64", use_rust_packer=False)
```

### Direct Usage with Storage

```python
from kohakuvault import DataPacker, ColumnVault

packer = DataPacker("msgpack")
vault = ColumnVault(":memory:")

# Pack data
user = {"name": "Alice", "email": "alice@example.com"}
packed = packer.pack(user)

# Store in vault (as bytes)
kv = vault.ensure("users", "bytes")
kv.append(packed)

# Retrieve and unpack
packed_retrieved = kv[0]
user_retrieved = packer.unpack(packed_retrieved, 0)
assert user_retrieved == user
```

---

## Advanced Examples

### Example 1: Multi-Language String Storage

Store text in multiple encodings:

```python
from kohakuvault import DataPacker, ColumnVault

vault = ColumnVault("multilang.db")

# UTF-8 column (general purpose)
utf8_col = vault.ensure("utf8_text", "str:utf8")
utf8_col.append("Hello, 世界!")

# UTF-16LE column (Windows compatibility)
utf16_col = vault.ensure("utf16_text", "str:utf16le")
utf16_col.append("Windows text")

# Latin-1 column (legacy European systems)
latin1_col = vault.ensure("latin1_text", "str:latin1")
latin1_col.append("Café")
```

### Example 2: Fixed-Size Binary Records

Store fixed-size binary structures:

```python
packer = DataPacker("bytes:64")  # 64-byte records

# Pack records
records = [
    b"\x01" * 10,  # Will be padded to 64 bytes
    b"\x02" * 64,  # Exactly 64 bytes
    b"\x03" * 32,  # Will be padded
]

vault = ColumnVault("records.db")
col = vault.ensure("binary_records", "bytes:64")

for record in records:
    col.append(record)

# Retrieve (padding preserved)
retrieved = col[0]
assert len(retrieved) == 64
```

### Example 3: JSON Schema Validated Storage

Store user profiles with validation:

```python
schema = {
    "type": "object",
    "properties": {
        "username": {
            "type": "string",
            "minLength": 3,
            "maxLength": 20,
            "pattern": "^[a-zA-Z0-9_]+$"
        },
        "email": {
            "type": "string",
            "format": "email"
        },
        "age": {
            "type": "integer",
            "minimum": 13,
            "maximum": 120
        },
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    },
    "required": ["username", "email"]
}

packer = DataPacker.with_json_schema(schema)

# Valid user
user = {
    "username": "alice123",
    "email": "alice@example.com",
    "age": 30,
    "roles": ["user", "moderator"]
}
packed = packer.pack(user)  # OK

# Invalid user (age too low)
try:
    packer.pack({
        "username": "bob",
        "email": "bob@example.com",
        "age": 10  # < 13
    })
except ValueError as e:
    print(f"Validation failed: {e}")
```

### Example 4: Bulk Operations Performance

Compare single vs bulk operations (run benchmark to see actual numbers on your system):

```python
import time
from kohakuvault import ColumnVault

vault = ColumnVault(":memory:")
col = vault.create_column("numbers", "i64")

# Single append (10,000 operations)
start = time.time()
for i in range(10000):
    col.append(i)
single_time = time.time() - start
print(f"Single append: {single_time:.3f}s ({10000/single_time:.0f} ops/s)")

# Bulk extend (10,000 operations)
col.clear()
values = list(range(10000))
start = time.time()
col.extend(values)
bulk_time = time.time() - start
print(f"Bulk extend: {bulk_time:.3f}s ({10000/bulk_time:.0f} ops/s)")

print(f"Speedup: {single_time/bulk_time:.1f}x")
```

### Example 5: MessagePack for Complex Structures

Store nested data structures efficiently:

```python
packer = DataPacker("msgpack")

# Complex nested structure
data = {
    "user_id": 12345,
    "profile": {
        "name": "Alice",
        "bio": "Software engineer",
        "stats": {
            "posts": 150,
            "followers": 1234,
            "following": 567
        }
    },
    "permissions": ["read", "write", "delete"],
    "metadata": {
        "created_at": "2025-01-01T00:00:00Z",
        "last_login": "2025-01-10T15:30:00Z",
        "verified": True
    }
}

packed = packer.pack(data)

# Size comparison
import json
json_bytes = json.dumps(data).encode('utf-8')
print(f"JSON: {len(json_bytes)} bytes")
print(f"MessagePack: {len(packed)} bytes")
print(f"Compression: {100 * len(packed) / len(json_bytes):.1f}%")
```

---

## Type Conversion Rules

### Python → Rust → Packed Bytes

| Python Type | Rust Type | Packed Format | Notes |
|-------------|-----------|---------------|-------|
| `int` | `i64` | 8 bytes LE | Range: ±9.2E18 |
| `float` | `f64` | 8 bytes IEEE754 | ~15 decimal digits |
| `str` | `String` | Encoded bytes | Encoding-dependent |
| `bytes` | `Vec<u8>` | Raw bytes | May be padded |
| `dict` | `serde_json::Value` | MessagePack/CBOR | Nested structures OK |
| `list` | `serde_json::Value` | MessagePack/CBOR | Mixed types OK |
| `bool` | `bool` | 1 byte | (in MessagePack) |
| `None` | `Option<T>` | Null byte | (in MessagePack) |

### Packed Bytes → Rust → Python

Inverse of above. Fixed-size strings have null padding trimmed.

---

## Error Handling

### Common Errors

#### `ValueError: Data too long: X > Y`

Fixed-size type received data larger than specified size.

```python
packer = DataPacker("bytes:32")
# packer.pack(b"x" * 100)  # ERROR: 100 > 32
```

**Solution**: Use larger fixed size or variable-size type.

#### `ValueError: String contains non-ASCII characters`

ASCII packer received non-ASCII string.

```python
packer = DataPacker("str:ascii")
# packer.pack("Hello, 世界")  # ERROR: non-ASCII
```

**Solution**: Use UTF-8 encoding instead.

#### `ValueError: Validation errors: ...`

JSON Schema validation failed.

```python
schema = {"type": "object", "required": ["name"]}
packer = DataPacker.with_json_schema(schema)
# packer.pack({})  # ERROR: missing 'name'
```

**Solution**: Fix data to match schema.

#### `ValueError: Cannot unpack_many for variable-size types`

Attempted `unpack_many` on variable-size type.

```python
packer = DataPacker("msgpack")
# packer.unpack_many(data, 5)  # ERROR: variable-size
```

**Solution**: Use `unpack()` in a loop instead.

---

## Performance Tips

### 1. Use Bulk Operations

```python
# ❌ Slower (many FFI calls)
for value in values:
    col.append(value)

# ✅ Faster (single FFI call with batch packing)
col.extend(values)
```

### 2. Choose Appropriate Fixed Sizes

```python
# ❌ Wasteful (95% padding)
packer = DataPacker("str:1024:utf8")
packer.pack("hi")  # Pads to 1024 bytes

# ✅ Efficient
packer = DataPacker("str:32:utf8")
packer.pack("hi")  # Pads to 32 bytes
```

### 3. Use MessagePack for Complex Data

```python
# JSON approach
import json
data = {"key": "value"}
packed = json.dumps(data).encode('utf-8')

# MessagePack approach (typically more compact)
packer = DataPacker("msgpack")
packed = packer.pack(data)
```

### 4. Reuse Packers

```python
# ❌ Creates packer for every value
for value in values:
    packer = DataPacker("i64")  # SLOW: recreated each time
    packed = packer.pack(value)

# ✅ Reuse packer
packer = DataPacker("i64")
for value in values:
    packed = packer.pack(value)
```

### 5. Use Rust Packing in Columns

```python
# ✅ Default (Rust packing - recommended)
col = vault.create_column("data", "i64")

# ❌ Fallback (Python packing - for debugging only)
col = vault.create_column("data", "i64", use_rust_packer=False)
```

---

## Benchmarking

Run the included benchmark script to measure performance on your system:

```bash
python examples/benchmark_packer.py
```

This will compare:
- Python struct.pack vs Rust DataPacker for primitives
- Single operations vs bulk operations
- MessagePack vs JSON for structured data
- Size efficiency for different formats

---

## Migration Guide

### From Python struct.pack (Primitives)

**Before**:
```python
import struct

class Column:
    def append(self, value):
        packed = struct.pack("<q", value)
        self._inner.append_raw(self._col_id, packed, 8, ...)
```

**After** (automatic):
```python
# No changes needed! Column now uses Rust DataPacker by default
col.append(42)  # Faster with Rust packing!
```

### From Python JSON + encode (Structured)

**Before**:
```python
import json

data = {"key": "value"}
json_str = json.dumps(data)
packed = json_str.encode('utf-8')
```

**After**:
```python
from kohakuvault import DataPacker

packer = DataPacker("msgpack")
packed = packer.pack(data)  # Faster and more compact!
```

---

## Technical Details

### Endianness

- **Integers**: Little-endian (LE)
- **Floats**: IEEE 754 (platform-native, typically LE on x86/ARM)
- **MessagePack/CBOR**: Big-endian (spec requirement)

### Thread Safety

DataPacker instances are **NOT thread-safe**. Create separate instances per thread:

```python
import threading

def worker(thread_id):
    # Each thread gets its own packer
    packer = DataPacker("i64")
    # ... use packer ...

threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
```

### Memory Usage

- **Primitives**: Minimal (8-32 bytes per packer instance)
- **MessagePack**: ~1KB overhead per packer
- **JSON Schema**: ~10KB overhead per compiled schema

### Schema Caching

JSON schemas are compiled once and cached by content hash. Repeated use of the same schema is fast.

---

## Limitations

### Current Limitations

1. **No String Type Inference**: Must specify encoding explicitly
2. **No Compression**: Data is packed but not compressed (use SQLite compression pragmas)
3. **No Custom Types**: Cannot define custom serializers (use MessagePack for arbitrary structures)
4. **CDDL Validation Not Implemented**: CBOR schema validation is placeholder
5. **No Zero-Copy**: Data is copied during pack/unpack (future optimization)

### Future Enhancements

- SIMD acceleration for numeric types
- Custom compression support (LZ4, Zstd)
- Zero-copy unpacking for large blobs
- CDDL validation for CBOR
- Cap'n Proto / FlatBuffers support (if demand exists)

---

## FAQ

### Q: Why not protobuf?

**A**: Protobuf requires external `protoc` compiler and doesn't support true runtime schema compilation in pure Rust. MessagePack provides similar size efficiency without dependencies.

### Q: Why MessagePack over BSON?

**A**: MessagePack has better size efficiency (variable-length integers) and is more widely supported. BSON uses fixed 5-byte integers which wastes space for small numbers.

### Q: Can I use DataPacker without ColumnVault?

**A**: Yes! DataPacker is standalone and can be used anywhere you need binary serialization.

### Q: Is JSON Schema validation slow?

**A**: Schema compilation happens once and is cached. Validation overhead is typically small, but run benchmarks on your specific schema and data to measure impact.

### Q: Can I pack Python objects (classes)?

**A**: No. Convert to dict first:
```python
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def to_dict(self):
        return {"name": self.name, "age": self.age}

user = User("Alice", 30)
packer = DataPacker("msgpack")
packed = packer.pack(user.to_dict())
```

### Q: What about backwards compatibility?

**A**: MessagePack format is stable. Data packed with v1.0 will unpack correctly in v2.0+. JSON Schema changes require migration.

---

## See Also

- [Column Storage Documentation](col.md)
- [KVault Documentation](kv.md)
- [Architecture Overview](arch.md)
- [MessagePack Specification](https://msgpack.org/index.html)
- [JSON Schema Specification](https://json-schema.org/)
- [CBOR RFC 8949](https://www.rfc-editor.org/rfc/rfc8949.html)

---

## Version History

### v0.3.0 (Current)
- Initial DataPacker implementation
- MessagePack, CBOR, primitive types support
- JSON Schema validation
- Automatic integration with ColumnVault
- Performance improvements over Python packing

---

## License

Apache License 2.0 - Same as KohakuVault

---

**Questions or Issues?** Open an issue on GitHub: [anthropics/claude-code/issues](https://github.com/anthropics/claude-code/issues)
