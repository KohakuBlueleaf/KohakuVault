# KohakuVault

SQLite-backed storage with Rust engine for high-performance key-value and columnar data.

## Quick Start

```bash
pip install kohakuvault
```

```python
from kohakuvault import KVault, ColumnVault, DataPacker

# Key-Value Storage
kv = KVault("data.db")
kv["user:123"] = b"user data"
kv["image:001"] = large_image_bytes

# Columnar Storage (shared DB)
cv = ColumnVault("data.db")
temps = cv.create_column("temperatures", "f64")
temps.extend([23.6, 23.8, 24.1])

profiles = cv.create_column("profiles", "msgpack")
profiles.append({"id": 1, "name": "Alice", "active": True})

# Bulk operations
scores = cv.create_column("scores", "i64")
scores.extend(list(range(100000)))  # 12.5M ops/s with cache

# Serialization
packer = DataPacker("msgpack")
packed = packer.pack({"key": "value"})
data = packer.unpack(packed, 0)
```

## Core Features

### KVault - Key-Value Storage

Dict-like interface for binary blobs:

```python
kv = KVault("vault.db")

# Basic operations
kv["key"] = b"value"
value = kv["key"]
del kv["key"]
"key" in kv

# Performance with caching
with kv.cache():
    for i in range(10000):
        kv[f"item:{i}"] = data

# Streaming for large files
with open("large_file.mp4", "rb") as f:
    kv.put_file("video:001", f)

with open("output.mp4", "wb") as f:
    kv.get_to_file("video:001", f)
```

### ColumnVault - Columnar Storage

List-like interface for typed columns:

```python
cv = ColumnVault("columns.db")

# Create columns with types
ids = cv.create_column("ids", "i64")
scores = cv.create_column("scores", "f64")
names = cv.create_column("names", "str:utf8")
metadata = cv.create_column("metadata", "msgpack")

# Append/extend
ids.append(1)
ids.extend([2, 3, 4, 5])

# Indexing and slicing
value = ids[0]
batch = ids[10:20]  # 200x faster than loop

# Update
ids[5] = 999
ids[10:15] = [100, 101, 102, 103, 104]
```

### DataPacker - Type-Safe Serialization

Efficient Rust-based serialization:

```python
packer = DataPacker("i64")
packed = packer.pack(42)           # 8 bytes
value = packer.unpack(packed, 0)   # 42

# Bulk operations (faster)
values = list(range(10000))
packed_all = packer.pack_many(values)
unpacked_all = packer.unpack_many(packed_all, count=10000)

# Supported types
"i64", "f64"                       # Primitives
"str:utf8", "str:32:utf8"         # Strings (variable/fixed)
"bytes", "bytes:128"               # Raw bytes
"msgpack", "cbor"                  # Structured data
```

## Performance

**M1 Max, 50K entries**:
- KVault write (64MB cache): ~24K ops/s @ 16KB (~377 MB/s)
- KVault read (hot cache): ~63K ops/s @ 16KB (~987 MB/s)
- Column extend (i64): ~12.5M ops/s with cache
- Column slice read (f64, 100 items): ~2.3M slices/s
- MessagePack: >1M ops/s

See `examples/benchmark.py` for reproducible benchmarks.

## Storage Interfaces

| Interface       | Model                   | Access        | Best For                      |
|-----------------|-------------------------|---------------|-------------------------------|
| `KVault`        | Key → value             | Dict-like     | Blobs, media, large files     |
| `Column`        | Fixed-size elements     | List-like     | Metrics, dense arrays         |
| `VarSizeColumn` | Variable-size elements  | List-like     | Logs, JSON, text              |
| `DataPacker`    | Type serializer         | Pack/unpack   | Custom pipelines              |
| `CSBTree`       | Ordered map             | Sorted access | Secondary indexes             |
| `SkipList`      | Ordered map             | Concurrent    | Lock-free access              |

## What's New in 0.7.0

### Auto-Packing (Enabled by Default)

Store any Python object without manual serialization:

```python
kv = KVault("data.db")

# Automatic!
kv["array"] = np.random.randn(768).astype(np.float32)  # numpy
kv["config"] = {"timeout": 30}                          # dict → MessagePack
kv["items"] = [1, 2, 3]                                 # list → MessagePack
kv["count"] = 42                                        # int
kv["image.jpg"] = jpeg_bytes                            # bytes (raw)

# Get returns actual objects!
config = kv["config"]  # dict, not bytes!
```

**Priority**: DataPacker types (i64, f64, vec:*) → str → MessagePack → Pickle fallback

### Vector Storage

Efficient array storage in columns:

```python
cv = ColumnVault("vectors.db")
embeddings = cv.create_column("text", "vec:f32:768")
images = cv.create_column("mnist", "vec:u8:28:28")

embeddings.extend([np.random.randn(768).astype(np.float32) for _ in range(1000)])
```

**Overhead**: 1 byte per vector

### Vector Similarity Search

k-NN search with sqlite-vec:

```python
from kohakuvault import VectorKVault

vkv = VectorKVault("search.db", dimensions=384, metric="cosine")
vkv.insert(embedding, doc.encode())
results = vkv.search(query_embedding, k=10)
```

**Metrics**: cosine, L2, L1, hamming

## Architecture

```
Python Layer
    ↓ PyO3 bindings
Rust Core (lib.rs)
    ↓ rusqlite + sqlite-vec
SQLite (single .db + WAL)
```

- **KVault**: Mutex-protected connection, write-back cache, BLOB API streaming
- **ColumnVault**: Chunked storage, adaptive sizing, batch operations
- **DataPacker**: Rust serializers (primitives, vectors, MessagePack, CBOR)
- **VectorKVault**: sqlite-vec integration, SIMD-accelerated search

All components share a single SQLite file.

## Advanced Features

- **Caching**: Write-back cache with configurable thresholds and auto-flush
- **Streaming**: BLOB API for large files without memory buffering
- **Batch operations**: Slice read/write, pack_many/unpack_many (3-35x faster)
- **Type wrappers**: `MsgPack(data)`, `Json(data)` for explicit encoding
- **Ordered containers**: CSBTree (B+Tree) and SkipList (lock-free)
- **Error handling**: Typed exceptions with automatic retry for busy states

## Development

```bash
pip install -e .[dev]
pytest                    # Run tests
cargo test                # Rust tests
cargo clippy             # Linting
```

## Documentation

- `examples/basic_usage.py` - Basic examples
- `examples/all_usage.py` - Comprehensive demos
- `docs/kv.md` - KVault details
- `docs/col.md` - ColumnVault details
- `docs/datapacker.md` - DataPacker types
- `docs/vectors.md` - Vector storage and search (v0.7.0)

## License

Apache License 2.0
