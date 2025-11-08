# Structured Data Cookbook

This page complements the [ColumnVault Guide](columnvault.md) with recipes focused on MessagePack/CBOR columns, schema validation, and hybrid layouts. Use it when you already know the basics of `ColumnVault` but need practical patterns.

## Choosing the Right Dtype

| Use case | Recommended dtype | Why |
|----------|-------------------|-----|
| JSON-like metadata with flexible keys | `msgpack` | Compact, schemaless, decoded transparently via `DataPacker`. |
| IoT / IETF interoperability | `cbor` or `cbor:256` | Standardized, supports more numeric types, optional schema validation. |
| Short human-readable tags | `str:32:utf8` | Fixed width keeps the column on the fast fixed-size path. |
| Mixed binary/text payloads | `bytes` + companion `msgpack` column | Keeps heavy blobs separate from structured metadata. |
| Deterministic record sizes | `msgpack:256`, `cbor:512` | Avoids `_idx` tables altogether by keeping every element the same width. |

## Schema Validation

`DataPacker` can enforce JSON Schema when serializing MessagePack payloads:

```python
from kohakuvault import ColumnVault, DataPacker

schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "name"],
}

packer = DataPacker.with_json_schema(schema)
vault = ColumnVault("users.db")
users = vault.ensure("users", "msgpack")
users.append({"id": 1, "name": "Rin"})
```

Failures raise `ValueError` before the write reaches SQLite, so consider validating in Python for hot loops and reserving schema enforcement for ingestion boundaries.

## Hybrid Layouts

### Metadata + Blobs

```python
from kohakuvault import KVault, ColumnVault

kv = KVault("media.db")
cols = ColumnVault(kv)
meta = cols.ensure("media_meta", "msgpack")

for asset in assets:
    kv[f"blob:{asset['id']}"] = asset["bytes"]
    meta.append({"id": asset["id"], "size": len(asset["bytes"]), "mime": asset["mime"]})
```

Binary payloads stay in `KVault`, while structured metadata lands in MessagePack so you can filter, slice, or extend without reparsing blobs.

### Secondary Indexes

```python
from kohakuvault import ColumnVault, CSBTree

cols = ColumnVault("events.db")
events = cols.ensure("events", "msgpack")
index = CSBTree()

row_id = len(events)
events.append({"user": 42, "type": "login"})
index.insert(42, row_id)

for _, rid in index.range(42, 42):
    handle(events[rid])
```

CSBTree/SkipList live inside the same PyO3 module, so you can build in-memory indexes keyed by user id, timestamp buckets, or composite keys while keeping the canonical record in a column.

## Performance Tips

- Batch operations (`extend`, slice assignment) beat Python loops—every call crosses the PyO3 boundary.
- Enable caches when appending many records: `with events.cache(cap_bytes=8 << 20): ...`.
- Tune `min_chunk_bytes`/`max_chunk_bytes` if you store large, variable payloads. Smaller chunks reduce rewrite cost when editing random rows.
- For large hot columns, occasionally call `ColumnVault.checkpoint()` to merge WAL changes and keep sqlite-vec (if you also use vectors) snappy.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ValueError: Invalid dtype` | Typo in dtype string | Use `DataPacker(dtype)` in a REPL to validate before creating the column. |
| `TypeError: Expected bytes` | Value not serializable under current dtype | Switch to a structured dtype (`msgpack`, `cbor`, etc.) or pre-serialize before appending (auto-pack wrappers only affect KVault). |
| Slice assignment raises `ValueError` | Replacement batch length mismatch | Ensure `len(values) == high - low`. |
| Updates are slow | Column is variable-size | Switch to fixed-size (`msgpack:NN`) where possible or increase chunk sizes. |

See [ColumnVault Guide](columnvault.md) for base APIs and [vectors.md](vectors.md) for vector-specific advice.
