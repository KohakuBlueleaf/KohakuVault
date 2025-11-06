# KVault: Key-Value Storage

`KVault` exposes a dict-like interface that streams binary blobs directly to SQLite using a Rust core. This guide walks through the data model, caching pipeline, retry behaviour, and recommended patterns.

## Conceptual Model

| Aspect | Details |
|--------|---------|
| Table schema | `CREATE TABLE kvault (key BLOB PRIMARY KEY, value BLOB NOT NULL)` |
| Backing API | `_KVault` (Rust), accessed via `kohakuvault.proxy.KVault` |
| Concurrency | Single writer, multiple readers via SQLite WAL |
| Keys | `bytes` or text (`str` encoded as UTF-8) |
| Values | Any bytes-like object (`bytes`, `bytearray`, `memoryview`, file streaming) |

## Basic Workflow

```python
from kohakuvault import KVault

vault = KVault("media.db")
vault["img:001"] = image_bytes           # setitem delegates to put()
assert vault["img:001"] == image_bytes   # getitem with retry handling

# Safe lookups
value = vault.get("missing", default=b"default")

# Streaming large blobs
with open("movie.mp4", "rb") as reader:
    vault.put_file("movie:2024", reader)
with open("movie_copy.mp4", "wb") as writer:
    vault.get_to_file("movie:2024", writer)
```

## Write-back Cache

KVault optimises bulk writes by buffering into a `WriteBackCache` (Rust) before flushing to SQLite.

```python
with KVault("logs.db").cache(cap_bytes=64 << 20, flush_threshold=16 << 20):
    for i in range(100_000):
        vault[f"log:{i}"] = gzip_blob(i)
```

Key behaviours:

- **Capacity guards**: any value larger than `cap_bytes` bypasses the cache and writes directly.
- **Flush triggers**: manual `flush_cache()`, threshold crossing, `disable_cache()`, cache context exit, or retry fallback on `CacheError::NeedFlush`.
- **Daemon support**: `enable_cache(..., flush_interval=5.0)` starts a Python daemon thread that flushes when no writes occur for `flush_interval` seconds.
- **Locking**: `vault.lock_cache()` toggles the `cache_locked` flag to defer auto flushes during multi-step operations.

## Streaming

Streaming operations avoid buffering entire blobs in Python:

| Method | Direction | Implementation detail |
|--------|-----------|-----------------------|
| `put_file(key, reader, size=None, chunk_size=None)` | Python -> SQLite | Streams via `put_stream_impl`, writing to a temporary zero-blob then `blob_open` |
| `get_to_file(key, writer, chunk_size=None)` | SQLite -> Python | Reads chunks via `blob_open` and writes to file-like object |

If `size` is omitted, `_file_size_of` tries to determine the remaining bytes using `seek/tell` or file descriptors; otherwise content is buffered into a `BytesIO`.

## Retry & Error Handling

All read/write methods call `_with_retries`, which:

1. Executes the underlying Rust closure.
2. On exception, maps it via `errors.map_exception`.
3. Retries `DatabaseBusy` up to `retries` times with exponential `backoff_base`.
4. Re-raises typed exceptions (`NotFound`, `InvalidArgument`, `IoError`, `KohakuVaultError`).

`KVault` obeys context manager semantics - `__enter__` returns `self`, `__exit__` flushes cached writes and checkpoints WAL before marking the vault as closed.

## API Reference

### Constructor

```python
KVault(
    path: str | os.PathLike,
    *,
    chunk_size: int = 1 << 20,
    retries: int = 4,
    backoff_base: float = 0.02,
    table: str = "kvault",
    enable_wal: bool = True,
    page_size: int = 32_768,
    mmap_size: int = 268_435_456,
    cache_kb: int = 100_000,
)
```

| Parameter | Effect |
|-----------|--------|
| `chunk_size` | Default chunk for streaming methods; actual cache threshold is separate. |
| `retries`, `backoff_base` | Control busy retry loop around all operations. |
| `table` | Allows hosting multiple logical stores in the same database. |
| `enable_wal`, `page_size`, `mmap_size`, `cache_kb` | Forwarded to SQLite pragmas in `common::open_connection`. |

### Operations

| Method | Description |
|--------|-------------|
| `put(key, value)` | Store bytes-like data. |
| `put_file(key, reader, size=None, chunk_size=None)` | Stream from file-like object. |
| `get(key, default=None)` | Return bytes or default value. |
| `__getitem__(key)` | Dict-style access that raises `KeyError` on missing key. |
| `get_to_file(key, writer, chunk_size=None)` | Stream content to a file-like object; returns bytes written. |
| `delete(key)` | Remove key; returns `True` on success. |
| `__delitem__(key)` | Dict-style delete (`KeyError` if missing). |
| `exists(key)` / `__contains__(key)` | Boolean membership checks. |
| `len(vault)` | Number of keys (SQL `COUNT(*)`). |
| `keys(prefix=None, limit=10_000)` | Iterator over keys (batched). |
| `values()` / `items()` | Iterators that load values lazily. |
| `enable_cache(...)` / `disable_cache()` / `flush_cache()` | Cache management. |
| `cache(..., auto_flush=True)` | Context manager wrapping `enable_cache` + optional flush. |
| `lock_cache()` | Context manager toggling the Rust-side lock flag. |
| `optimize()` | Runs `PRAGMA optimize; VACUUM;`. |
| `checkpoint()` | Forces WAL checkpoint (safe to call while open). |
| `close()` | Flushes cache, checkpoints WAL, stops daemon thread, marks instance closed. |

## Interop with ColumnVault

Pass a `KVault` instance to `ColumnVault` to share the underlying database file:

```python
kv = KVault("analytics.db")
cols = ColumnVault(kv)  # uses kv._path internally

kv["blob:123"] = large_blob
metrics = cols.ensure("metrics", "f64")
```

This pattern enables hybrid designs where binary payloads reside in `KVault` while metadata and analytics live in columnar storage.

## Maintenance Patterns

- **Manual checkpointing**: For write-heavy workloads, call `checkpoint()` periodically to keep WAL size manageable.
- **Graceful shutdown**: Always `close()` or use context managers to flush caches and stop daemon threads.
- **Large values**: When caching, ensure `cap_bytes` exceeds the largest expected value; otherwise writes will bypass the cache (safe but slower).
- **Backups**: Because everything is stored in a single SQLite file, copying `media.db` while no writer is active is a safe backup strategy.

## Exceptions

| Python exception | Origin |
|------------------|--------|
| `KohakuVaultError` | Generic catch-all for unexpected runtime errors. |
| `DatabaseBusy` | SQLite busy/locked after exhaustive retries. |
| `NotFound` | Lookup misses in Rust (`query_row` errors) mapped to Python. |
| `InvalidArgument` | Type/coercion errors raised in Python validation or DataPacker. |
| `IoError` | File I/O failures when streaming to/from file-like objects. |

`errors.map_exception` provides a single conversion point; custom tooling can reuse it when wrapping other PyO3 calls.

## Testing & Benchmarks

- Unit tests: `tests/test_basic.py`, `tests/test_setitem.py`, and related suites cover basic CRUD, caching, concurrency tolerance, and combined workflows.
- Benchmarks: `examples/benchmark.py` exercises configurable data sizes, cache configurations, and streaming throughput.

Understanding `KVault`'s data flow and cache strategy sets the foundation for combining it with columnar and structured storage components provided by KohakuVault.
