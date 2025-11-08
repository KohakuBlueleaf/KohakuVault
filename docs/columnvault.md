# ColumnVault Guide

`ColumnVault` exposes list-like typed columns stored in the same SQLite database as `KVault`. Each column tracks its dtype, chunk sizes, and caches in Rust so Python code can treat it like a `MutableSequence` without worrying about serialization or chunk math.

## Concepts at a Glance

| Component | Description |
|-----------|-------------|
| `_ColumnVault` | Rust backend that owns the SQLite connection, chunk allocator, and per-column caches. |
| `Column` | Fixed-size column proxy (ints, floats, fixed strings/bytes, fixed MessagePack/CBOR, fixed vector shapes). |
| `VarSizeColumn` | Variable-size proxy backed by paired `{name}_data` + `{name}_idx` tables storing chunk offsets. |
| `DataPacker` | Serializer/deserializer for dtype strings. Python falls back to `DTYPE_INFO` when the extension is missing. |
| Cache Daemon | Optional background thread that flushes buffered appends after an idle interval. |

## Creating and Discovering Columns

```python
from kohakuvault import ColumnVault
import numpy as np

vault = ColumnVault("analytics.db")
counts = vault.create_column("counts", "i64")
counts.extend(range(10))

# Reuse existing column or create it lazily
metrics = vault.ensure("metrics", "msgpack")
```

`ColumnVault.list_columns()` returns `(name, dtype, length)` triplets; `ColumnVault.delete_column()` removes data plus metadata.

## DType Grammar

`create_column(name, dtype, **opts)` accepts the full DataPacker grammar:

| Category | Examples | Notes |
|----------|----------|-------|
| Integers | `"i64"`, `"i32"` | Stored little-endian; see `pack_i64`/`pack_i32`. |
| Floats | `"f64"`, `"f32"` | IEEE-754 doubles/floats. |
| Fixed bytes | `"bytes:32"` | Zero-padded/truncated to fixed length. |
| Variable bytes | `"bytes"` | Stored verbatim; offsets tracked in `_idx`. |
| Strings | `"str:utf8"`, `"str:32:utf8"`, `"str:utf16le"`, `"str:latin1"`, `"str:ascii"` | Fixed-width forms pad/truncate encoded bytes. |
| Structured | `"msgpack"`, `"msgpack:256"`, `"cbor"`, `"cbor:512"` | Prefer fixed sizes (`:N`) when payloads have an upper bound to stay on the fixed-size fast path. |
| Vectors | `"vec:f32:768"`, `"vec:u8:3:224:224"`, `"vec:f32"` | Fixed-shape (`vec:<type>:dim[:dim...]`) or arbitrary-shape (`vec:<type>`). Supported element types: `f32`, `f64`, `i8/16/32/64`, `u8/16/32/64`. |

`parse_dtype()` tries to instantiate a Rust `DataPacker` first and falls back to Python helpers (`DTYPE_INFO`) if the extension isn’t available.

## Fixed-Size Columns (`Column`)

- Appends/extends buffer values in Rust, aligning chunk boundaries to `elem_size`.
- `__getitem__` supports integers and slices; slices stream batches to minimise Python overhead.
- `__setitem__` accepts scalars or slices—slice assignments pack the entire batch before calling `_setitem_slice` so there’s only one round-trip to Rust.
- Iteration reads 1000-element batches for speed.
- `insert`/`delete` are supported but incur chunk rewrites when the change isn’t at the tail; design for append-heavy workloads when possible.

```python
scores = vault.create_column("scores", "f64")
scores.extend([95.5, 87.3, 99.1])
scores[1:3] = [90.0, 91.5]
```

## Variable-Size & Structured Columns (`VarSizeColumn`)

Structured data is just a var-size column with a dtype that knows how to (de)serialize values:

```python
events = vault.ensure("events", "msgpack")
events.append({"type": "login", "user": 42})
events.extend({"type": "purchase", "amount": a} for a in amounts)

# JSON Schema validation with DataPacker
from kohakuvault import DataPacker
schema = {"type": "object", "required": ["id", "name"]}
packer = DataPacker.with_json_schema(schema)
```

Implementation details:

- Appends choose between typed (`append_typed[_cached]`) and raw bytes depending on whether a `DataPacker` exists.
- Reads look up `(chunk_id, start, end)` from `{name}_idx` before slicing bytes. With a `DataPacker`, `batch_read_varsize_unpacked` returns decoded Python objects.
- Updates handle three cases: equal size (in-place overwrite), shrink (overwrite + shift), grow (rebuild chunk and adjust offsets). All logic happens in Rust to avoid Python loops.
- Slice assignments use `update_varsize_slice`, so you can replace arbitrary segments atomically.

## Vector Columns

Use vector dtypes when you need dense tensors but don’t need search yet:

```python
embeddings = vault.create_column("embeddings", "vec:f32:768")
embeddings.extend(np.random.randn(10_000, 768).astype(np.float32))

images = vault.create_column("mnist", "vec:u8:28:28")
```

Fixed-shape vectors add only 1 byte of overhead (type byte). Arbitrary-shape vectors encode `ndim` and shape metadata (`2 + ndim * 4` bytes) before the raw buffer.

Combine with `DataPacker("vec:f32:768")` when you need to batch `pack_many`/`unpack_many` operations outside `ColumnVault`.

## Caching & Chunk Tuning

```python
with vault.cache(cap_bytes=32 << 20, flush_threshold=8 << 20):
    counts.extend(range(1_000_000))

col = vault["counts"]
with col.cache(cap_bytes=4 << 20):
    col.append(123)
```

- `ColumnVault.cache()` toggles caches for every known column, great for coordinated ingests.
- Individual columns also expose `cache()`/`enable_cache()`/`flush_cache()` APIs.
- `lock_cache()` exists on both the vault and column proxies to delay daemon flushes during multi-step operations.
- Override `min_chunk_bytes`/`max_chunk_bytes` in `create_column` if defaults (128 KiB / 16 MiB) don’t match your workload. Smaller chunks help high-churn tables; bigger chunks benefit sequential scans.

## Patterns

- **Metadata + Blobs** – pair `ColumnVault` for structured metadata with `KVault` for binary payloads.
- **Secondary Indexes** – combine `ColumnVault` columns with `CSBTree`/`SkipList` when you need ordered lookups on derived keys.
- **Hybrid Analytics** – store IDs in `i64`, timestamps in `i64`, embeddings in `vec:f32:384`, and metadata in `msgpack` while pointing everything at the same SQLite file.

## Error Handling & Testing

Operations raise `InvalidArgument` for dtype mismatches, `NotFound` for missing columns, and `KohakuVaultError` for runtime issues. Relevant tests live under `tests/test_columnar.py`, `tests/test_structured_columns.py`, `tests/test_varsize_operations.py`, and `tests/test_column_cache.py`.

For more structured-data recipes, see `docs/structured_columns.md`. For vector-specific advice (including `VectorKVault`), jump to `docs/vectors.md`.
