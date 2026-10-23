# KohakuVault

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KohakuBlueleaf/KohakuVault)

One-file SQLite datastore with:
  * `KVault` key-value storage inspired by [BoringDB](https://github.com/mel-project/boringdb) (streaming blobs + auto-pack for dict/list/numpy)
  * `TextVault` full-text search with FTS5 BM25 ranking (ideal for RAG pipelines)
  * `ColumnVault` buffer-oriented arrays, tensors, and variable-width records using bounded SQLite BLOB blocks
  * write-back caches plus standalone CSB+Tree / SkipList containers for ordered/temporal metadata
  * `VectorKVault` (sqlite-vec) for k-NN search living in the same `.db`


```python
from kohakuvault import KVault, TextVault, ColumnVault, VectorKVault, CSBTree
import numpy as np

DB = "test.db"

# Store media + metadata in one file
kv = KVault(DB, table="media")
kv["video:42"] = b"abcdefg" # raw bytes stay raw
kv["video:42:meta"] = {"fps": 60, "tags": ["tutorial", "gpu"]}  # auto MessagePack

# One atomic committed KV batch; reads preserve input order
kv.put_many([("run:1", {"score": 0.9}), ("run:2", {"score": 0.8})])
assert kv.get_many(["run:2", "missing"])[0] == {"score": 0.8}

# Full-text search with BM25 ranking (great for RAG)
tv = TextVault(DB, table="docs")
tv.insert("Machine learning is a subset of AI", {"category": "tech"})
for doc_id, score, meta in tv.search("machine learning", k=5):
    print(f"Doc {doc_id}: score={score:.4f}")

# Incremental columnar logging with bounded append buffers
cols = ColumnVault(kv)  # shares the database file, not the KV transaction
ids = cols.ensure("ids", "i64")
latency = cols.ensure("latency_ms", "f64")
embeddings = cols.ensure("embeddings", "vec:f32:384")
with cols.cache(cap_bytes=8 << 20):
    ids.append(len(ids))
    latency.append(12.3)
    embeddings.append(np.random.randn(384).astype(np.float32))

# Ordered metadata via embedded CSBTree
index = CSBTree()
index.insert(latency[-1], ids[-1])  # log-latency -> run-id

# Vector similarity search (sqlite-vec inside the same DB)
search = VectorKVault(DB, table="runs", dimensions=384, metric="cosine")
for idx, emb in enumerate(embeddings[:10]):
    search.insert(emb, str(idx).encode())
for rank, (rid, distance, run_id) in enumerate(search.search(embeddings[-1], k=3), 1):
    print(rank, distance, run_id.decode())

search.close()
cols.close()
kv.close()
```


## Why this exists
- **Embedded reader/writer access** – SQLite WAL permits independent readers alongside one active writer, without a database service. Runtime `-wal`/`-shm` sidecars are normal; use a consistent backup rather than copying a live main file alone.
- **Buffer-oriented columns** – Fixed-width arrays and tensor batches stay contiguous. Variable records use payload/offset blocks. Small reads fetch the requested BLOB ranges instead of materializing whole chunks.
- **Complete iteration** – KV iteration is unlimited by default and paginated internally. `limit` is an optional total cap; `batch_size` controls query size.
- **Auto-packed KV everywhere** – `KVault` (BoringDB-inspired) automatically serializes dict/list/numpy values and keeps raw bytes untouched; every key-value interaction goes through this pipeline.
- **Write caches and native batches** – Vault- and column-level caches batch inserts; `put_many()` commits an explicit KV batch atomically. Pending/in-flight cache reads do not wait for SQLite flush I/O.
- **Standalone CSB+Tree / SkipList** – Specialized containers for ordered metadata or range queries without leaving the process.
- **Vector keys/search** – `vec:*` dtypes store tensors, and `VectorKVault` (sqlite-vec) runs k-NN search inside the same DB.

If your workflow looks like *“keep logging forever, but don’t spawn thousands of files”*, KohakuVault is the tool.

## Installation

```bash
pip install kohakuvault
```

For development (build the PyO3 extension, run tests):

```bash
pip install -e .[dev]

# Rebuild after changing native code; use release mode for performance tests.
maturin develop --release
```

## Use cases / references
- **Media vaults** – Random-access blobs + metadata (`tests/test_kv_headers.py`, `examples/basic_usage.py`).
- **RAG pipelines** – Full-text BM25 search + vector similarity in the same DB (`tests/test_text_vault.py`).
- **Incremental ML logging** – Fixed/var columns, vector dtypes, caches (`tests/test_columnar.py`, `examples/columnar_demo.py`).
- **Vector-heavy retrieval** – `VectorKVault`, CSBTree/SkipList, auto-packed metadata (`tests/test_vector_kvault.py`, `examples/vector_search_numpy.py`, `examples/all_usage.py`).

## Performance notes

Use `append_array()`, `read_array()`, and `iter_batches()` for numeric/tensor throughput. Scalar/list APIs remain convenient but include Python object conversion costs.

```bash
python examples/benchmark_columnar.py --rows 1000000
python examples/benchmark_kv.py --structured --repeats 5
python examples/benchmark_readers.py
python examples/kvault_demo.py
```

A local release-build comparison reduced 1,000 small i64 slices from 7.61 seconds to 29.94 milliseconds by removing repeated whole-BLOB materialization. Other measured gains were smaller: about 2.1x for list append and 2.3x for list read. These are single-run, warm-process results, not universal throughput guarantees. See [method and measurements](docs/performance.md).

**Storage compatibility:** KV/vector formats are preserved. The new column format does not read or migrate the legacy `col_*` layout. See [column storage design](docs/columnar_storage.md).

## Documentation

- [Changes and migration notes](docs/changes.md) – breaking behavior, new APIs, and compatibility
- [Documentation index](docs/README.md)
- [KVault Guide](docs/kvault.md)
- [TextVault Guide](docs/textvault.md) – Full-text search with BM25 ranking
- [ColumnVault Guide](docs/columnvault.md)
- [Column Storage Design](docs/columnar_storage.md)
- [Performance Checks](docs/performance.md)
- [Vector Storage & Search](docs/vectors.md)
- [DataPacker Reference](docs/datapacker.md)
- [Architecture Deep Dive](docs/architecture.md)
- [Structured Data Cookbook](docs/structured_columns.md)

## Development workflow

```bash
pip install -e .[dev]
maturin develop
```

Formatting/linting/testing before PR:
```bash
black .
cargo fmt
cargo clippy
maturin develop --release
pytest
```

- PyO3 module: `src/kvault-rust` (use `maturin develop` if you prefer).
- Format Python with `black`.
- Keep docs updated and linked via `docs/README.md`.

## License

Apache 2.0. See [LICENSE](LICENSE).
