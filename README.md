# KohakuVault

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KohakuBlueleaf/KohakuVault)

One-file SQLite datastore with:
  * `KVault` key-value storage inspired by [BoringDB](https://github.com/mel-project/boringdb) (streaming blobs + auto-pack for dict/list/numpy)
  * `TextVault` full-text search with FTS5 BM25 ranking (ideal for RAG pipelines)
  * `ColumnVault` columnar layout inspired by Stanchion (typed chunks: `i64`, `msgpack`, `vec:*`) built on SQLite BLOB ranges
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

# Full-text search with BM25 ranking (great for RAG)
tv = TextVault(DB, table="docs")
tv.insert("Machine learning is a subset of AI", {"category": "tech"})
for doc_id, score, meta in tv.search("machine learning", k=5):
    print(f"Doc {doc_id}: score={score:.4f}")

# Incremental columnar logging (no file rewrites)
cols = ColumnVault(kv) # or ColumnVault(DB)
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
```


## Why this exists
- **Single-file SWMR** – SQLite WAL lets one writer and many readers share a file. Parquet can’t append, DuckDB locks the DB, Lance spawns version files.
- **Columnar layout, not SQL tables** – `ColumnVault` stores chunks in `col_chunks`/`_idx`, Stanchion-style, so incremental appends don’t rewrite tables.
- **Auto-packed KV everywhere** – `KVault` (BoringDB-inspired) automatically serializes dict/list/numpy values and keeps raw bytes untouched; every key-value interaction goes through this pipeline.
- **Write caches** – Vault- and column-level caches batch inserts automatically.
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

# everytime you change the rust code, you should run this command
maturin develop
```

## Use cases / references
- **Media vaults** – Random-access blobs + metadata (`tests/test_kv_headers.py`, `examples/basic_usage.py`).
- **RAG pipelines** – Full-text BM25 search + vector similarity in the same DB (`tests/test_text_vault.py`).
- **Incremental ML logging** – Fixed/var columns, vector dtypes, caches (`tests/test_columnar.py`, `examples/columnar_demo.py`).
- **Vector-heavy retrieval** – `VectorKVault`, CSBTree/SkipList, auto-packed metadata (`tests/test_vector_kvault.py`, `examples/vector_search_numpy.py`, `examples/all_usage.py`).

## Performance notes

Benchmarks on an M1 Max (WAL on, cache enabled):
- `KVault` streaming + cache: **24K writes/s**, **63K reads/s** (`examples/benchmark.py`).
- `ColumnVault` extend (`i64`): **12.5M ops/s**; slicing 100 elements hits **2.3M slices/s**.
- `DataPacker` vector unpack: **35× faster** than Python loops for 768-dim tensors.

## Documentation

- [Documentation index](docs/README.md)
- [KVault Guide](docs/kvault.md)
- [TextVault Guide](docs/textvault.md) – Full-text search with BM25 ranking
- [ColumnVault Guide](docs/columnvault.md)
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
