# KohakuVault

SQLite-backed storage with vector search, auto-packing, and high-performance Rust engine.

## Quick Start

```bash
pip install kohakuvault
```

### Auto-Packing (New in 0.7.0!)

```python
from kohakuvault import KVault
import numpy as np

kv = KVault("data.db")  # Auto-pack enabled by default!

# Store any Python object - automatic serialization!
kv["embedding"] = np.random.randn(768).astype(np.float32)
kv["config"] = {"timeout": 30, "enabled": True}
kv["scores"] = [95.5, 87.3, 92.1]
kv["count"] = 42

# Get returns actual objects, not bytes!
config = kv["config"]  # dict
embedding = kv["embedding"]  # numpy array
```

### Vector Similarity Search (New in 0.7.0!)

```python
from kohakuvault import VectorKVault

vkv = VectorKVault("search.db", dimensions=384, metric="cosine")

# Index documents
for doc, embedding in zip(documents, embeddings):
    vkv.insert(embedding, doc.encode())

# Search
results = vkv.search(query_embedding, k=10)
```

### Vector Columns (New in 0.7.0!)

```python
from kohakuvault import ColumnVault

cv = ColumnVault("vectors.db")
embeddings = cv.create_column("text_embeddings", "vec:f32:768")
embeddings.extend([np.random.randn(768).astype(np.float32) for _ in range(1000)])
```

## Storage Interfaces at a Glance

| Interface            | Data model              | Access pattern        | Backing tables / structures            | Highlights                                         | Best for                               |
|----------------------|-------------------------|-----------------------|----------------------------------------|----------------------------------------------------|----------------------------------------|
| `KVault`             | Key -> any Python object | Dict-style            | `kvault` (blob)                        | **Auto-packing**, streaming, retry logic, caching  | Any data type, media files, ML models  |
| `VectorKVault`       | Vector -> value          | k-NN similarity search| `vec0` virtual table (sqlite-vec)      | **Fast vector search**, multiple metrics (cosine/L2) | Semantic search, recommendations     |
| `Column`             | Fixed-size elements     | Mutable sequence      | `col_meta` + `col_chunks`              | **Vector storage**, batch slice ops, Rust packing  | Embeddings, images, dense arrays       |
| `VarSizeColumn`      | Prefixed variable bytes | Mutable sequence      | `{name}_data` + `{name}_idx`           | Size-aware updates, adaptive chunk growth          | Logs, JSON payloads, text              |
| `DataPacker`         | Typed serializer        | Pack/unpack helpers   | Pure Rust (no extra tables)            | **Vector types**, MessagePack/CBOR, 35x faster bulk | Custom pipelines, preprocessing        |
| `CSBTree`            | Ordered map             | B+Tree style API      | Arena-backed cache-sensitive tree      | Contiguous nodes, iterator & range queries         | Sorted secondary indexes, metadata     |
| `SkipList`           | Ordered map             | Lock-free (CAS)       | Lock-free skip list                    | Concurrent inserts/reads without GIL contention    | Shared read/write heaps, hot paths     |

## Capabilities

- **Auto-packing (v0.7.0)**: Store numpy arrays, dicts, lists, primitives without manual serialization. Automatic encoding/decoding.
- **Vector search (v0.7.0)**: Fast k-NN similarity search with sqlite-vec. Cosine, L2, L1, hamming metrics.
- **Vector storage (v0.7.0)**: Efficient array/tensor storage in columns. Minimal 1-byte overhead for fixed-shape vectors.
- **Optimized bulk ops (v0.7.0)**: 35x faster pack_many/unpack_many for vectors using numpy.stack/frombuffer.
- Rust-powered I/O with Python-first ergonomics (PyO3 bridge).
- Write-back cache for both key-value and columnar workloads (context manager, daemon auto-flush, capacity guards).
- Fast range access: `Column.__getitem__` batches reads; slice assignment funnels to Rust.
- Variable-size column maintenance: prefix-sum index, chunk rebuilds, and fragment tracking.
- Concurrency aware retry logic that turns SQLite busy states into typed exceptions.
- Optional CSB+Tree and SkipList implementations for ordered access patterns.

## Architecture

```
Python layer (proxy.py / column_proxy.py)
    -> PyO3 bindings
Rust core (lib.rs)
    -> rusqlite + custom allocators
SQLite storage (single .db + WAL)
```

- **KVault**: mutex-protected connection, optional write-back cache, streaming via BLOB API.
- **ColumnVault**: element-aligned chunking, cache buckets per column, adaptive variable-size slices.
- **DataPacker**: Rust serializers report `elem_size` / `is_varsize` to Python, enabling automatic dtype strategy.
- **SkipList / CSBTree**: share Python key wrappers to support arbitrary `PyObject` ordering.

## Performance Snapshot

### New in 0.7.0
- **Vector bulk unpack**: 35x faster than loop (10K 768-dim embeddings)
- **Primitives bulk**: 2.66x faster pack, 5.92x faster unpack (1M values)
- **MessagePack**: 42% smaller than JSON, >800K ops/s
- **Vector overhead**: 1 byte for fixed-shape (0.19% for 768-dim f32)

### Existing (M1 Max, 50K entries)
- KVault write with 64 MiB cache: ~24k ops/s at 16 KiB payloads (~377 MB/s).
- KVault read hot cache: ~63k ops/s at 16 KiB payloads (~987 MB/s).
- Column `extend` (`i64`): ~12.5M ops/s with cache, >450x faster than uncached.
- Column slice read (`f64`, 100 items): ~2.3M slices/s, 200x faster than per-element.

See `examples/benchmark.py` and `examples/benchmark_datapacker_new.py` for benchmarks.

## Tooling & Extras

- **Auto-packing (v0.7.0)**: Automatic serialization/deserialization for numpy arrays, dicts, lists, primitives. MessagePack for dicts/lists (efficient!), DataPacker for arrays. Disabled with `kv.disable_auto_pack()`.
- **Vector types (v0.7.0)**: `vec:f32:768` (fixed), `vec:i64:10:20` (2D), `vec:u8:3:224:224` (RGB), `vec:f32` (arbitrary). 10 element types: f32, f64, i32, i64, u8, u16, u32, u64, i8, i16.
- **VectorKVault (v0.7.0)**: Vector similarity search with k-NN queries. Supports cosine, L2, L1, hamming metrics. Built on sqlite-vec.
- **Type wrappers (v0.7.0)**: `MsgPack(data)`, `Json(data)`, `Cbor(data)`, `Pickle(data)` for explicit encoding control.
- **DataPacker**: primitives, strings, vectors, `msgpack`, `cbor`, JSON Schema validation. `pack_many`/`unpack_many` with numpy optimization.
- **Write-back cache**: `vault.cache(...)` and `column.cache(...)` with auto-flush, daemon threads, capacity guards.
- **Mixed workloads**: KVault, ColumnVault, and VectorKVault share single SQLite file.
- **Ordered indexes**: `CSBTree` (cache-sensitive B+Tree) and `SkipList` (lock-free).
- **Error mapping**: `RuntimeError` → `KohakuVaultError`, `DatabaseBusy`, `NotFound`, `InvalidArgument`, `IoError`.

## Development & Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Format and lint
ruff check --fix .
black src tests examples
cargo fmt

# Run Python + Rust tests
pytest
cargo test
```

The repository uses maturin for building the extension; see `pyproject.toml` for configuration.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
