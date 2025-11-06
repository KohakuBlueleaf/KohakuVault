# Structured Columns Guide

ColumnVault can store complex Python objects (dicts, lists, strings) using the bundled `DataPacker`. This guide focuses on best practices when working with structured and variable-size data.

## Getting Started

```python
from kohakuvault import ColumnVault

vault = ColumnVault("events.db")
events = vault.ensure("events", "msgpack")

events.append({"type": "login", "user": 42})
events.extend(
    {"type": "purchase", "user": 42, "amount": 19.99}
    for _ in range(10)
)

latest = events[-1]
assert latest["type"] == "purchase"
```

`msgpack` and `cbor` dtypes automatically create a `DataPacker`; elements are packed/unpacked transparently in Rust.

## Choosing a Format

| Format | Use when | Pros | Cons |
|--------|----------|------|------|
| `msgpack` | General-purpose structured data | Compact, fast, schema-less | No enforced schema |
| `cbor` | IETF compliance or IoT compatibility | Standardized, self-describing | Larger than MessagePack for some payloads |
| `str:utf8` (variable) | Human-readable strings | Easy debugging | Requires manual parsing for complex structures |
| Fixed strings (`str:32:utf8`) | Uniform short keys | Predictable chunk alignment | Truncation if over limit |

For heterogeneous records with optional fields, MessagePack tends to offer the best balance.

## Column Layout

Structured columns are variable-size columns under the hood:

```
events_data (dtype stored as "msgpack")  -- raw bytes
events_idx  (adaptive_idx)               -- chunk id + start/end offsets
```

`VarSizeColumn` handles:
- Appends using `append_typed[_cached]`.
- Slice reads via `batch_read_varsize_unpacked`.
- Updates using size-aware operations (`update_varsize_element`, `update_varsize_slice`).

## JSON Schema Validation

Use `DataPacker.with_json_schema` when you need strict schemas:

```python
from kohakuvault import ColumnVault, DataPacker

schema = {"type": "object", "required": ["id", "name"]}
packer = DataPacker.with_json_schema(schema)

vault = ColumnVault("users.db")
col = vault.ensure("users", "msgpack")

col.append({"id": 1, "name": "Maki"})           # OK
col.append({"id": 2})                           # ValueError
```

For performance-critical paths, consider validating before writing to avoid runtime failures in hot loops.

## Hybrid Patterns

### Metadata + Blobs

```python
kv = KVault("media.db")
cols = ColumnVault(kv)

meta = cols.ensure("media_meta", "msgpack")

for asset in assets:
    kv[f"blob:{asset['id']}"] = asset["bytes"]
    meta.append({"id": asset["id"], "size": len(asset["bytes"])})
```

Metadata remains queryable in `meta`, while the actual binary lives in KVault.

### Secondary Indexes with CSBTree

```python
from kohakuvault import CSBTree

events = cols.ensure("events", "msgpack")
index = CSBTree()

row_id = len(events)
events.append({"id": row_id, "user": 42})
index.insert(42, row_id)

for _, rid in index.range(42, 42):
    handle(events[rid])
```

Use CSBTree or SkipList when you need ordered lookups by a derived key.

## Performance Tips

- **Batch operations**: prefer `extend` and slice assignments over per-element loops.
- **Cache wisely**: enable column caches when appending large batches. The `cache()` context manager flushes automatically.
- **Slice size**: `VarSizeColumn.__getitem__` supports slices, batch reading in Rust; use this instead of iterating per element when scanning windows.
- **Chunk sizing**: adjust `min_chunk_bytes`/`max_chunk_bytes` when creating the column if the default 128 KiB/16 MiB does not match your workload.
- **Fixed-width structured types**: when payloads have a bounded size, choose dtypes like `msgpack:256` or `cbor:512` so the column can operate in the fixed-size fast path; truly variable-size structured columns rely on prefix-sum indexes and are slower to mutate.

## Error Handling

- Updating with incompatible types raises `TypeError`.
- JSON Schema violations raise `ValueError`.
- Accessing missing structured columns raises `errors.NotFound`.
- Slice length mismatches (`col[a:b] = values`) raise `ValueError` before calling into Rust.

## Testing

- `tests/test_structured_columns.py` covers MessagePack/CBOR round trips, slice operations, and cache paths.
- `tests/test_varsize_operations.py` ensures size-aware updates behave correctly.
- `examples/datapacker_demo.py` and `examples/benchmark_container.py` illustrate real-world structured workloads.

## When Not to Use Structured Columns

- When the schema is extremely wide and better suited to a document database.
- When you need ad-hoc field-level queries - KohakuVault does not index inside MessagePack payloads.
- When multi-writer concurrency is required; SQLite WAL still enforces a single writer at a time.

Structured columns shine when you need append-friendly storage with automatic serialization in a single-file deployment.
