# ColumnVault: Columnar Storage

`ColumnVault` delivers list-like access to typed columns inside the same SQLite database as `KVault`. It supports fixed-width primitives, adaptive variable-size data, and optional Rust-based packing.

## Core Concepts

| Component | Description |
|-----------|-------------|
| `_ColumnVault` | Rust backend managing chunk allocation, caches, and vacuum helpers. |
| `Column` | Python proxy for fixed-size columns (implements `MutableSequence`). |
| `VarSizeColumn` | Python proxy for variable-size columns with prefix-sum indexes. |
| Column metadata | `col_meta` records id, dtype string, element size, logical length, min/max chunk size. |
| Chunks | `col_chunks` stores payload data per column, one blob per chunk. |
| Index column | `{name}_idx` stores 12-byte descriptors for variable-size elements (chunk id, start, end). |

## DType Grammar

`ColumnVault.create_column(name, dtype, ...)` accepts the following patterns:

| Category | Examples | Notes |
|----------|----------|-------|
| Integers | `"i64"` | 8-byte little-endian signed integers. |
| Floats | `"f64"` | 8-byte IEEE-754 doubles. |
| Fixed bytes | `"bytes:32"` | Zero-padded to specified length. |
| Variable bytes | `"bytes"` | Stored as raw bytes in `_data`, offsets tracked in `_idx`. |
| Strings | `"str:utf8"`, `"str:32:utf8"`, `"str:utf16le"`, `"str:latin1"` | Uses `DataPacker` when available. Fixed-size forms pad/truncate. |
| Structured | `"msgpack"`, `"msgpack:256"`, `"cbor"`, `"cbor:256"` | MessagePack or CBOR serialized by `DataPacker`; prefer fixed-size variants (`msgpack:NN`, `cbor:NN`) for fastest columnar operations. |

`ColumnVault` invokes `parse_dtype()`, which attempts to instantiate `DataPacker`. When present, it defers dtype logic to Rust; otherwise it falls back to legacy helpers defined in `column_proxy.py`.

Fixed-size structured dtypes (for example `msgpack:256` or `cbor:512`) keep each element the same width, allowing columns to stay on the fixed-size fast path. Purely variable-size structured columns are supported, but they rely on prefix-sum indexes and are noticeably slower for heavy mutation workloads.

## Creating & Accessing Columns

```python
from kohakuvault import ColumnVault

vault = ColumnVault("analytics.db")

counts = vault.create_column("counts", "i64")
counts.extend(range(10))
assert counts[0] == 0
assert counts[2:5] == [2, 3, 4]

events = vault.create_column("events", "msgpack")
events.append({"kind": "login", "user": 123})
assert events[0]["user"] == 123
```

- `ColumnVault.ensure(name, dtype, ...)` returns an existing column or creates a new one when absent.
- Access via `vault[name]`; missing columns raise `errors.NotFound`.
- Sharing with `KVault`: pass a `KVault` instance instead of a path to reuse its SQLite connection file.

## Fixed-size Columns (`Column`)

- Stored directly in `col_chunks`, aligned to element size to avoid cross-chunk fragmentation.
- `append(value)` and `extend(values)` optionally use `DataPacker` for packing.
- `__getitem__` implements:
  - index access with chunk-local reads (`read_range`).
  - slice access that batches elements via `batch_read_fixed`.
- `__setitem__` accepts scalars or slices; slice assignment leverages `_setitem_slice` to pack all values and call a single `write_range`.
- Iteration reads in 1000-element batches for efficiency.
- `delete`, `insert` trigger chunk rewrites; deleting from the end simply shrinks the recorded length.

## Variable-size Columns (`VarSizeColumn`)

- Created via `_create_varsize_column` which spawns `{name}_data` (dtype stored in metadata) and `{name}_idx`.
- Index entries hold (chunk id, start byte, end byte); data is stored in chunk-local offsets.
- `append(value)` chooses between typed append (`append_typed`) and raw bytes depending on packer availability.
- Reads resolve chunk and byte range before returning: if a Rust `DataPacker` is available and the dtype is structured, `batch_read_varsize_unpacked` returns decoded objects.
- `__setitem__` handles three size cases: shrink-in-place, equal size, or grow requiring chunk rebuild via `update_varsize_element`.
- Slice assignment uses `update_varsize_slice`, performing batched updates entirely in Rust.

## Caching

Each column can enable a write-back cache layered atop `_ColumnVault`:

```python
with counts.cache(cap_bytes=8 << 20, flush_threshold=2 << 20):
    for value in heavy_series:
        counts.append(value)
```

- `Column.enable_cache` / `VarSizeColumn.enable_cache` wrap Rust cache APIs.
- `ColumnVault.enable_cache` toggles caching for all known columns and optionally starts a flush daemon.
- `ColumnVault.cache(...)` manages caches for all columns, useful when populating multiple columns in sync.
- `ColumnVault.lock_cache()` exposes the Rust lock flag to defer auto flushes during critical sections.

## Maintenance

- `ColumnVault.flush_cache()` flushes all column caches through Rust.
- `ColumnVault.checkpoint()` triggers WAL checkpointing on the shared connection.
- `ColumnVault.delete_column(name)` removes metadata and data entries; caches are invalidated.
- `ColumnVault.list_columns()` returns `(name, dtype, length)` tuples from metadata.
- `ColumnVault.enable_cache(..., flush_interval)` spins up a Python daemon thread that periodically checks `get_cache_idle_time` on the Rust side.

## API Cheatsheet

| Method | Description |
|--------|-------------|
| `create_column(name, dtype, chunk_bytes=None, use_rust_packer=True)` | Create a new fixed or variable-size column. |
| `ensure(name, dtype, ...)` | Get existing column or create new column lazily. |
| `__getitem__(name)` | Retrieve `Column` or `VarSizeColumn`. |
| `list_columns()` | Introspect existing columns with dtype and length. |
| `delete_column(name)` | Remove a column (and paired `_data`/`_idx` if present). |
| `enable_cache(...)` / `disable_cache()` / `flush_cache()` / `cache(...)` | Vault-level cache controls. |
| `lock_cache()` | Context manager to temporarily block daemon flushes. |

### Column methods (selected)

| Category | Methods |
|----------|---------|
| Mutation | `append`, `extend`, `insert`, `__setitem__`, `__delitem__` |
| Access | `__getitem__`, `__iter__`, `__len__` |
| Cache | `enable_cache`, `disable_cache`, `flush_cache`, `cache`, `lock_cache` |
| Utilities | `__repr__`, `checkpoint`, `flush_cache` (vault) |

## Patterns

### Hybrid Metadata + Blobs

```python
kv = KVault("library.db")
cols = ColumnVault(kv)

ids = cols.ensure("book_ids", "i64")
titles = cols.ensure("titles", "str:utf8")
blobs = kv  # reuse kv for binary content

for book in library:
    ids.append(book.id)
    titles.append(book.title)
    blobs[f"book:{book.id}"] = book.pdf_bytes
```

Metadata queries stay in-column while the original binary lives in KVault.

### Structured Records

```python
events = cols.ensure("events", "msgpack")
events.append({"event": "login", "user_id": 42, "ip": "127.0.0.1"})
for event in events[-100:]:
    if event["event"] == "error":
        handle(event)
```

DataPacker automatically serializes/deserializes the Python dicts; no manual encoding required.

### Variable-size Updates

```python
logs = cols.ensure("logs", "bytes")
logs.append(b"short")
logs.append(b"longer entry")

logs[1] = b"replacement"  # size-aware update in Rust
logs[0:2] = [b"a"*5, b"b"*12]  # batched slice update
```

Chunk rebuilds happen transparently when new data outgrows original chunks.

## Error Handling

Operations raise:

- `InvalidArgument` when dtype-specific constraints fail (e.g., string encoding mismatch).
- `NotFound` when referencing missing columns.
- `KohakuVaultError` for unexpected runtime errors.

VarSize mutations sanitise byte lengths; Python ensures provided slices have matching lengths before delegating to Rust.

## Testing

Relevant pytest suites:

- `tests/test_columnar.py` - fundamental operations, slicing, cache contexts.
- `tests/test_dynamic_chunks.py` - chunk growth and resizing.
- `tests/test_varsize_operations.py` - variable-size append, slice setitem.
- `tests/test_structured_columns.py` - `msgpack`/`cbor` flows.
- `tests/test_column_cache.py` - vault- and column-level caching scenarios.

Understanding these components enables building analytics, logs, and metadata-heavy workloads on top of KohakuVault's columnar subsystem.
