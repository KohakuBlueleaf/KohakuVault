# Structured Data Cookbook

This page complements the [ColumnVault Guide](columnvault.md) with recipes focused on MessagePack/CBOR columns, schema validation, and hybrid layouts. Use it when you already know the basics of `ColumnVault` but need practical patterns.

## Choosing the Right Dtype

| Use case | Recommended dtype | Why |
|----------|-------------------|-----|
| JSON-like metadata with flexible keys | `msgpack` | Compact, schemaless, decoded transparently via `DataPacker`. |
| IoT / IETF interoperability | `cbor` or `cbor:256` | Standardized, supports more numeric types, optional schema validation. |
| Short human-readable tags | `str:32:utf8` | Fixed width keeps the column on the fast fixed-size path. |
| Mixed binary/text payloads | `bytes` + companion `msgpack` column | Keeps heavy blobs separate from structured metadata. |
| Deterministic record sizes | `msgpack:256`, `cbor:512` | Uses fixed-width payload blocks without variable-record offsets. |

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
with ColumnVault("users.db") as vault:
    users = vault.ensure("validated_users", "bytes")
    # Explicitly validate and encode before storing the serialized record.
    users.append(packer.pack({"id": 1, "name": "Rin"}))
    record = packer.unpack(users[0], 0)
```

The explicit `packer.pack()` call enforces the schema before the write reaches SQLite. Creating a separate validating packer does not automatically attach it to a `msgpack` column. This recipe stores already-validated bytes and decodes them explicitly.

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

CSBTree/SkipList are independent in-memory indexes. Their contents are not persisted or committed atomically with column writes. Close the column vault explicitly when finished.

## Performance Tips

- Prefer `extend()` for structured ingestion and `append_array()`/`read_array()` for numeric or fixed-shape tensor batches. Structured records still require per-record serialization.
- Enable caches when appending many records: `with events.cache(cap_bytes=8 << 20): ...`.
- Tune `block_bytes` for append and range-read workloads. Variable-width replacements currently rebuild the entire column; changing the block size does not remove that cost.
- Use explicit `close()` or a context manager. `checkpoint()` reports WAL checkpoint progress, but a long-running reader can prevent full progress; it is not a general search-speed optimization.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ValueError: Invalid dtype` | Typo in dtype string | Check the type table in the ColumnVault guide; column numeric codecs and standalone DataPacker do not support identical type sets. |
| `TypeError: Expected bytes` | Value not serializable under current dtype | Switch to a structured dtype (`msgpack`, `cbor`, etc.) or pre-serialize before appending (auto-pack wrappers only affect KVault). |
| Slice assignment raises `ValueError` | Replacement batch length mismatch | Ensure `len(values) == high - low`. |
| Updates are slow | Column is variable-size | Prefer append-heavy use, move frequently mutated records to KV, or use fixed-width records with a suitable maximum size. |

See [ColumnVault Guide](columnvault.md) for base APIs and [vectors.md](vectors.md) for vector-specific advice.
