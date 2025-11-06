# DataPacker Reference

`DataPacker` is the Rust serializer/deserializer bundled with KohakuVault. It powers columnar dtype parsing and is available as a standalone utility for custom pipelines.

## Features

- Zero-copy interface in Rust; exposed to Python via PyO3.
- Primitive support (`i64`, `f64`), flexible byte encodings (`bytes`, `bytes:N`).
- Rich string handling (`str:utf8`, `str:utf16le`, `str:utf16be`, `str:latin1`, `str:ascii`, fixed-width variants).
- Structured formats (`msgpack`, `cbor`) with optional JSON Schema validation.
- Batch APIs (`pack_many`, `unpack_many`) to avoid Python loops.
- Metadata via `elem_size` and `is_varsize` for downstream chunk planning.

## Creating a Packer

```python
from kohakuvault import DataPacker

packer = DataPacker("str:32:utf8")
assert packer.elem_size == 32
assert not packer.is_varsize
```

If the dtype denotes a variable-size format, `elem_size` returns `0` and `is_varsize` becomes `True`.

### Supported Dtype Strings

| Category | Pattern | Notes |
|----------|---------|-------|
| Integer | `i64` | 8-byte signed integer |
| Float | `f64` | 8-byte double |
| Bytes (fixed) | `bytes:N` | Zero-padded to `N` bytes |
| Bytes (variable) | `bytes` | Stored verbatim |
| Strings (variable) | `str:utf8`, `str:utf16le`, `str:utf16be`, `str:latin1`, `str:ascii` | UTF encodings validated; ASCII throws on non-ASCII characters |
| Strings (fixed) | `str:N:utf8`, `str:N:utf16le`, etc. | Padded or truncated to `N` encoded bytes |
| MessagePack | `msgpack` or `msgpack:fixed=N` (optional fixed size) | Serialises dicts/lists |
| CBOR | `cbor` | CBOR encoding via `ciborium` |

For column-heavy workloads, prefer fixed-size structured dtypes such as `msgpack:256` or `cbor:512`. Keeping every element the same width avoids the slower variable-size code paths driven by prefix-sum indexes.

Invalid dtype strings raise `ValueError`.

## Packing & Unpacking

```python
packer = DataPacker("msgpack")
payload = {"user": 1, "tags": ["vip", "beta"]}

blob = packer.pack(payload)
assert isinstance(blob, bytes)

decoded = packer.unpack(blob, 0)
assert decoded["tags"][0] == "vip"
```

Offset semantics:

- `pack()` returns a `bytes`.
- `unpack(data, offset=0)` reads a single value from `data` starting at byte offset.

## Batch Operations

### Fixed-size Types

```python
packer = DataPacker("i64")
values = [1, 2, 3, 4]

blob = packer.pack_many(values)
assert len(blob) == len(values) * packer.elem_size
assert packer.unpack_many(blob, count=len(values)) == values
```

### Variable-size Types

```python
packer = DataPacker("str:utf8")
names = ["Rin", "Nozomi", "Hanayo"]

blob = packer.pack_many(names)      # concatenated bytes
offsets = []
cursor = 0
for name in names:
    encoded = name.encode("utf-8")
    offsets.append(cursor)
    cursor += len(encoded)

restored = packer.unpack_many(blob, offsets=offsets)
```

When calling `unpack_many`:

- Pass `count=...` for fixed-size types.
- Pass `offsets=[...]` for variable-size types.
- Do not mix both parameters.

## JSON Schema Validation

MessagePack packers can enforce JSON Schema rules:

```python
schema = {
    "type": "object",
    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
    "required": ["id", "name"],
}

packer = DataPacker.with_json_schema(schema)
valid = {"id": 1, "name": "Rin"}
packer.pack(valid)             # OK
packer.pack({"id": "oops"})    # Raises ValueError
```

Compiled schemas are cached by content hash (MD5) to avoid recompilation overhead.

## Integration with ColumnVault

`column_proxy.Column` and `VarSizeColumn` attempt to create a `DataPacker` using the column dtype. Effects:

- Automatic detection of `elem_size`/`is_varsize` guides chunk alignment.
- `Column` operations use `pack`, `pack_many`, `unpack`, `unpack_many` when available.
- `VarSizeColumn` leverages `pack_many` and `batch_read_varsize_unpacked` for structured types, falling back to raw bytes if the packer is unavailable.
- If the compiled extension is missing, Python lambdas from `DTYPE_INFO` provide legacy behaviour.

## Error Semantics

| Exception | Trigger |
|-----------|---------|
| `ValueError` | Invalid dtype string, schema violation, insufficient data during unpack. |
| `TypeError` | Passing incompatible value types (e.g., non-bytes to `bytes:N`). |

Rust errors are converted to Python `ValueError`/`TypeError` before crossing the FFI boundary.

## Performance Notes

- Creating a `DataPacker` is cheap; caching instances per dtype avoids repeated validation.
- For structured data, avoid re-serializing offsets in Python - store offsets alongside data (e.g., save to a separate column).
- JSON Schema validation introduces overhead proportional to schema complexity; keep schemas minimal in hot paths.

## Threading

Packer instances are not thread-safe. In multi-threaded Python code, instantiate one packer per thread:

```python
def worker():
    packer = DataPacker("i64")
    ...
```

## Testing & Examples

- `tests/test_packer.py` covers primitives, strings, structured types, and column integration.
- `examples/benchmark_packer.py` compares DataPacker with pure Python packing and demonstrates combined workflows.
- `examples/datapacker_demo.py` showcases schema validation, structured columns, and bulk operations.

Use these resources to evaluate performance and validate dtype choices in your workload.
