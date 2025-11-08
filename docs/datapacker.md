# DataPacker Reference

`DataPacker` is the Rust serializer/deserializer shared by `ColumnVault`, `KVault`’s auto-pack pipeline, and standalone workflows. It validates dtype strings, packs values without Python loops, and exposes batch APIs for bulk transfers.

## Supported Dtype Strings

| Category | Pattern | Notes |
|----------|---------|-------|
| Integers | `i8`, `i16`, `i32`, `i64` | Little-endian; booleans map to `i8`. |
| Floats | `f32`, `f64` | IEEE-754. |
| Bytes | `bytes`, `bytes:N` | Variable or fixed-width zero-padded buffers. |
| Strings | `str:utf8`, `str:utf16le`, `str:utf16be`, `str:latin1`, `str:ascii`, `str:N:utf8`, etc. | Fixed-width versions pad/truncate encoded bytes. |
| Structured | `msgpack`, `msgpack:N`, `cbor`, `cbor:N` | Pick fixed sizes when possible to keep `ColumnVault` in the fixed-size fast path. |
| Vectors | `vec:<elem>[:dim[:dim...]]` | `<elem>` ∈ `{f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}`. Without dims → arbitrary shape (stores ndim + shape). |

Creating a packer validates the dtype immediately:

```python
from kohakuvault import DataPacker

fixed = DataPacker("str:32:utf8")
assert fixed.elem_size == 32 and not fixed.is_varsize

var = DataPacker("vec:f32")
assert var.elem_size == 0 and var.is_varsize
```

## Packing & Unpacking

```python
packer = DataPacker("msgpack")
blob = packer.pack({"user": 1, "tags": ["vip", "beta"]})
restored = packer.unpack(blob, offset=0)
```

Offsets only matter for buffer views—you can store multiple values in one blob and read them by passing different offsets.

## Batch APIs

```python
from kohakuvault import DataPacker
import numpy as np

pack_i64 = DataPacker("i64")
values = list(range(1_000))
buffer = pack_i64.pack_many(values)
assert pack_i64.unpack_many(buffer, count=len(values)) == values

pack_vec = DataPacker("vec:f32:768")
vectors = [np.random.randn(768).astype(np.float32) for _ in range(1_000)]
buffer = pack_vec.pack_many(vectors)
restored = pack_vec.unpack_many(buffer, count=len(vectors))
```

- Fixed-size types use `count=...`.
- Variable-size types use `offsets=[...]` (accumulate byte positions yourself).
- Mixing `count` and `offsets` raises `ValueError`.

## JSON Schema Validation

```python
schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string"},
    },
    "required": ["id", "name"],
}

packer = DataPacker.with_json_schema(schema)
packer.pack({"id": 1, "name": "Rin"})
packer.pack({"id": "oops"})  # raises ValueError
```

Schemas are cached by MD5 hash, so reusing the same schema in multiple packers is cheap.

## Auto-Pack Integration

`KVault.enable_auto_pack()` instantiates `DataPacker` internally to serialize numpy arrays (`vec:*`), ints (`i64`), floats (`f64`), strings, and structured data. Wrappers (`MsgPack`, `Json`, `Cbor`, `Pickle`) are optional—they set `encoding_name` to override the automatic choice when you need a specific format.

If the Rust extension isn’t available, python fallbacks in `column_proxy.py` handle legacy dtypes (`i64`, `f64`, `bytes:N`). Prefer installing the extension for performance and full dtype coverage.

## Threading & Performance

- Packer instances are **not** thread-safe—create one per thread if you plan to reuse them in worker pools.
- Construction is cheap; cache packers per dtype to avoid repeated validation.
- JSON Schema validation and arbitrary-shape vectors allocate temporary buffers proportional to the payload; fixed-size packers can stream directly into preallocated buffers.

## Testing & Examples

- `tests/test_packer.py` covers primitives, strings, structured payloads, and vector types.
- `examples/datapacker_demo.py` shows schema validation + `ColumnVault` integration.
- `examples/benchmark_packer.py` compares `pack_many`/`unpack_many` against pure Python loops.

Keep this page in sync with new dtype strings—`ColumnVault`, auto-pack, and future APIs all rely on DataPacker to stay consistent.
