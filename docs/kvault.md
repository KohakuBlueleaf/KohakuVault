# KVault Guide

`KVault` is the dict-like face of KohakuVault. It runs on top of a Rust core (`_KVault`) that streams bytes directly into SQLite, adds an auto-pack pipeline for Python objects, and layers caching/streaming helpers around it.

## Why Use It?
- Works like a Python `dict`, but stores data durably in SQLite.
- Auto-pack (enabled by default) turns numpy arrays, dicts, lists, and scalars into compact binary automatically.
- Write-back caches and streaming APIs let you ingest millions of records without Python loops becoming the bottleneck.
- Sharing the same `.db` file as `ColumnVault`/`VectorKVault` keeps deployments simple.

## Quick Start

```python
from kohakuvault import KVault
import numpy as np

kv = KVault("app.db")  # auto-pack is on by default
kv["config"] = {"timeout": 30, "retries": 3}         # auto MessagePack
kv["embedding"] = np.random.randn(768).astype(np.float32)
kv["image.jpg"] = open("image.jpg", "rb").read()     # raw bytes stay raw

assert kv["config"]["timeout"] == 30
assert kv["embedding"].shape == (768,)
```

## Auto-Pack and Headers

Auto-pack is implemented in Rust (`kv/autopacker.rs`) and follows this priority:

1. `bytes` → stored as-is, no header.
2. Explicit wrappers (`MsgPack`, `Json`, `Cbor`, `Pickle`) → force a particular encoding when you need it.
3. NumPy arrays → `vec:*` DataPacker dtype.
4. Integers/floats → `i64`/`f64` DataPacker dtype.
5. `str` → UTF-8.
6. dict/list → MessagePack with optional Pickle fallback.
7. Everything else → Pickle if enabled.

The resulting blob may include a 10-byte header (`|0x89 0x4B|version|encoding|flags|reserved|0x56 0x4B|`) so reads can auto-detect encoding. You control behaviour with:

```python
kv.enable_auto_pack(use_pickle=True)   # default
kv.disable_auto_pack()                 # revert to bytes-only
kv.auto_pack_enabled()                 # -> bool
```

Need raw headers for custom tooling? `enable_headers()`/`disable_headers()` toggle the header injection while keeping bytes untouched.

### Explicit encodings (optional)

Wrappers from `kohakuvault.wrappers` are only needed when you must override the automatic choice. For example, wrap a value in `Json(...)` if an external system expects JSON bytes or `MsgPack(...)` if you want to guarantee MessagePack even when auto-pack would normally choose Pickle. Plain dicts/lists/numpy arrays do **not** require wrappers.

## Data Model & Defaults

| Aspect | Value |
|--------|-------|
| Table name | `kv` (override via `KVault(..., table="...")`) |
| Schema | `CREATE TABLE kv (key BLOB PRIMARY KEY, value BLOB NOT NULL)` |
| Key types | `bytes` or `str` (UTF-8 encoded) |
| Value types | Any auto-packable Python object, bytes-like objects, file-like streams |
| Chunk size | `chunk_size` arg (1 MiB default) controls stream batch size |
| SQLite PRAGMAs | WAL on, 32 KiB page size, 256 MiB mmap, cache 100 MB by default |

## Caching and Concurrency

`KVault` uses a Rust write-back cache so you can buffer millions of writes before hitting SQLite:

```python
with kv.cache(cap_bytes=64 << 20, flush_threshold=8 << 20) as buffered:
    for i in range(200_000):
        buffered[f"log:{i}"] = b"..."

kv.enable_cache(flush_interval=5.0)  # background daemon flush
with kv.lock_cache():                 # block daemon flush temporarily
    mutate_many_keys()
kv.flush_cache()
```

- Values larger than `cap_bytes` bypass the cache automatically.
- `flush_cache()` is safe to call repeatedly; it only flushes dirty entries.
- `lock_cache()` is a context manager that defers auto-flushes when you need deterministic batching.
- Every operation is wrapped in `_with_retries` (exponential backoff, 4 attempts by default) to deal with SQLite `BUSY`/`LOCKED` cases.

## Streaming APIs

```python
# Upload large blobs without loading them fully in memory
with open("movie.mp4", "rb") as reader:
    kv.put_file("movie:2024", reader, chunk_size=4 << 20)

with open("copy.mp4", "wb") as writer:
    kv.get_to_file("movie:2024", writer, chunk_size=4 << 20)
```

`put_file` writes through the SQLite incremental BLOB API (zero-blob, `blob_open`); `get_to_file` streams in chunks. When `size` is omitted, KohakuVault attempts to determine it via `seek/tell` or file descriptors; otherwise it buffers into a `BytesIO` fallback.

## API Cheat Sheet

| Category | Methods |
|----------|---------|
| CRUD | `__getitem__`, `__setitem__`, `__delitem__`, `get`, `put`, `delete`, `exists`, `__contains__`, `len` |
| Iteration | `keys(prefix=None, limit=10_000)`, `values()`, `items()` (all batched) |
| Streaming | `put_file`, `get_to_file` |
| Caching | `enable_cache`, `disable_cache`, `flush_cache`, `cache(cap_bytes, flush_threshold, auto_flush=True)`, `lock_cache()` |
| Auto-pack | `enable_auto_pack`, `disable_auto_pack`, `auto_pack_enabled`, `enable_headers`, `disable_headers`, `headers_enabled` |
| Maintenance | `checkpoint`, `optimize`, `close` |

All methods raise typed exceptions from `kohakuvault.errors`: `DatabaseBusy`, `NotFound`, `InvalidArgument`, `IoError`, or `KohakuVaultError` for unexpected failures.

## Interop Patterns

- **Share the DB with ColumnVault** – pass a `KVault` instance to `ColumnVault(kv)` to reuse its database file when mixing blobs and typed columns.
- **Metadata + Blobs** – keep large binary payloads in `KVault` while storing structured metadata in MessagePack columns.
- **Vector Pipelines** – auto-pack stores numpy arrays directly; pair this with `VectorKVault` for similarity search when you need indexing.

## Maintenance & Diagnostics

- Call `kv.checkpoint()` in long-running writers to keep the WAL small.
- `kv.optimize()` issues `PRAGMA optimize; VACUUM;`—handy after bulk deletes.
- Always `close()` or use context managers so caches flush and WAL checkpoints run, otherwise background daemon threads keep the process alive.
- For 0.6.x compatibility, `disable_auto_pack()` returns you to the old bytes-only behaviour instantly.

## Testing & Benchmarks

- Pytests: `tests/test_basic.py`, `tests/test_setitem.py`, `tests/test_cache.py` cover CRUD, caching, retries, and streaming.
- Benchmarks: `examples/benchmark.py` shows cache vs non-cache throughput; `examples/all_usage.py` exercises hybrid KVault/ColumnVault/VectorKVault workflows.

Keep this guide handy when adding new methods—if it changes behaviour, document it here and link from `docs/index.md` so users know where to look.
