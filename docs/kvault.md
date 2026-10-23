# KVault Guide

`KVault` is the dict-like face of KohakuVault. It runs on top of a Rust core (`_KVault`) that streams bytes directly into SQLite, adds an auto-pack pipeline for Python objects, and layers caching/streaming helpers around it.

## Why Use It?
- Works like a Python `dict`, but stores data durably in SQLite.
- Auto-pack (enabled by default) turns numpy arrays, dicts, lists, and scalars into compact binary automatically.
- Write-back caches batch writes; streaming APIs avoid loading large blobs at once.
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

## Native bulk operations

```python
kv.put_many([("config", {"enabled": True}), ("blob", b"raw")])
values = kv.get_many(["blob", "missing", "config"], default=None)
```

`put_many()` accepts an explicit list or tuple of key/value pairs, serializes the complete batch before database work, and commits one atomic transaction. Duplicate keys use the last value; an invalid earlier duplicate still fails validation. The call bypasses the write-back cache and removes matching pending values only after commit. Other pending keys remain buffered. A locked enabled cache or a read-only handle rejects the call, including an empty batch. Failed serialization or SQL leaves storage and pending values unchanged.

`get_many()` accepts a list or tuple of keys and returns values in input order, including duplicates. Missing keys return the supplied default; stored `None` remains `None`. All committed lookups share one SQLite snapshot; matching pending/in-flight references are captured first and take precedence. The cache capture and database snapshot are not one atomic view under concurrent mutation. Decoding occurs after the snapshot is released.

These APIs materialize the entire explicitly supplied batch. They reject generators instead of silently consuming unbounded input. Split large inputs yourself; separately committed batches are not one atomic operation. Neither bulk method automatically retries Python serialization or database errors. `update(other=(), **kwargs)` consumes mappings or iterables in batches of at most 1024 pairs, delegating to `put_many()`. It commits directly instead of buffering, applies keyword pairs last, and is atomic per batch, not for the whole iterable. Earlier successful batches survive a later input/serialization/SQL failure. Use explicit scalar puts to retain cache-buffered acceptance.

## Mapping operations

- `pop(key)` atomically removes and returns the decoded value, raising `KeyError` if absent. `pop(key, default)` returns that exact default only for a missing key; a stored `None` is a value. Reads/deletes share an immediate transaction, and pending values are removed only after successful SQL. The deletion commits before decoding, so a decoding exception does not restore the raw record.
- `setdefault(key, default=None)` returns the existing decoded value without serializing an unused default. If absent, it serializes outside locks, then rechecks under the writer reservation and inserts atomically. Insertions commit directly and reject an enabled cache lock; existing pending values remain readable. Non-byte defaults are no longer coerced through `bytes()`.
- `clear()` issues one transactional `DELETE` and retires this handle's pending data after commit. A trigger/SQL failure leaves all existing and pending values intact. Like `delete()`/`pop()`, it commits directly even inside a cache context.
- Mutating mapping calls reject read-only handles; `setdefault()` can return an already-existing value on a read-only handle. `pop()`, `setdefault()` and `clear()` do not automatically replay on failure.

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

- Values larger than `cap_bytes` bypass the cache unless an enabled cache is locked; then the write raises `DatabaseBusy` before changing storage. A successful bypass supersedes the earlier pending value for that key.
- `flush_cache()` only clears dirty entries after its transaction commits. A failed flush leaves the batch cached and readable so the caller can retry; `flush_cache()` and `disable_cache()` do not automatically retry failures.
- `disable_cache()` flushes and removes the cache as one synchronized operation. Concurrent successful writes survive the transition. Calling it while `lock_cache()` is active raises `DatabaseBusy` and keeps the cache enabled; retry after leaving the context.
- Nested `lock_cache()` contexts suppress publication of new flush batches until the last context exits. An already in-flight batch may still commit; entering the context does not wait for or undo it. A flush waiting for the writer connection rechecks the flag before publishing a batch. This is not a database transaction and does not prevent other callers from writing. A full locked cache raises `DatabaseBusy`; leave the context and flush before retrying.
- Capacity counts key/value bytes in both pending and in-flight batches, not total process memory or container overhead. Cache-pressure writers wait for the serialized flush or receive its error. Readers can inspect immutable in-flight values while SQLite is busy. Failed flush recovery keeps newer pending replacements rather than restoring old values over them.
- `close()` inside a cache-lock context raises `DatabaseBusy` and leaves the handle usable. Successful close fences mutation admission and flushes before releasing the native handle; a failed final flush remains retryable. Cache contexts flush on exit and are not rollback transactions.
- Scalar `put()`, `get()`, `delete()`, and `exists()` retry transient SQLite errors with exponential backoff (4 total attempts by default). Successful point operations avoid allocating retry closures. `retries` must be positive; backoff must be finite and nonnegative.
- Background flush failure emits a `RuntimeWarning`, stores the exception in `background_flush_error`, stops that daemon, and retains pending writes. Resolve the cause and call `flush_cache()` explicitly; success while unlocked clears the recorded error. Call `enable_cache(flush_interval=...)` again if background flushing should resume. Intervals must be finite and positive; no errors are silently discarded.

Database operations release the Python GIL around native connection/SQLite waits.
For asyncio applications, call these synchronous methods through a worker, for
example `await asyncio.to_thread(kv.put, key, value)`. The write still waits for
SQLite's writer lock, but that wait does not block unrelated Python threads.
A cache transaction holds the writer connection, but not the cache-state mutex,
through SQLite waits and commit. Same-handle committed and buffered reads can
therefore proceed while a file-backed flush is blocked by another writer.
Python value conversion and streaming reader/writer callbacks still require the
GIL; this does not make CPU-heavy Python serialization asynchronous. Uncontended memory-only cache admission uses nonblocking lock attempts without a GIL round trip; any wait/SQL work still detaches. Serializer configuration is copied under a short mutex and released before Python conversion, so serializer/deserializer callbacks can access the same vault without deadlocking its packer mutex. Such callback side effects are independent operations, not part of the outer write's rollback.

## Complete iteration and read-only access

```python
with KVault("app.db", read_only=True) as reader:
    for key in reader:  # no 10,000-key cutoff
        print(key)
    for key, value in reader.items(prefix="log:", batch_size=128):
        consume(key, value)
    first_hundred = list(reader.keys(limit=100))
```

`limit=None` means no total cap. `batch_size` controls query size, not the total number yielded. `items()` fetches values alongside keys rather than running an extra query for every value. Batches are row-bounded, not byte-bounded; use smaller batches for large values. Prefix scans support arbitrary bytes, including empty and all-0xff prefixes.

Iteration uses keyset pagination with a fresh snapshot per page. It is not a single immutable snapshot under concurrent mutation. The current handle's pending cache entries are included; another handle sees them only after commit. Cached overwrites replace the existing pending value, and deleting a pending key does not resurrect it at flush.

`read_only=True` opens an existing database without schema writes or journal-mode changes. File-backed point reads and scans use independent read-only connections so they need not share the writer's connection mutex. SQLite still permits one active write transaction. File-backed streaming reads also use independent snapshots. `reader_cache_kb=8192` sets each KV reader's SQLite cache budget; readers inherit `mmap_size`. `reader_idle_limit=8` limits retained idle connections, not active readers; zero disables idle retention. Memory databases and SQLite URI paths use the primary connection instead. `reader_pool_stats()` reports `idle_connections` and `sqlite_cache_bytes` (SQLite page-cache allocations for idle readers only, not total process RSS, mmap residency, or active-reader memory). Point queries use the SELECT statement's snapshot rather than an extra BEGIN/ROLLBACK; bulk, length-overlay and streaming multi-step reads still use explicit snapshots.

The KV table format, headers, and auto-pack encodings are unchanged by the column rewrite. Use `close()` or a context manager for deterministic release of native connections. A failed cache flush leaves the vault open for retry.

## Streaming APIs

```python
# Upload large blobs without loading them fully in memory
with open("movie.mp4", "rb") as reader:
    kv.put_file("movie:2024", reader, chunk_size=4 << 20)

with open("copy.mp4", "wb") as writer:
    kv.get_to_file("movie:2024", writer, chunk_size=4 << 20)
```

`put_file` allocates and fills the incremental BLOB inside one transaction. Short input, invalid callback output, or callback exceptions roll back the entire stream: an old value remains intact, and a new partial key is not published. Successful stream writes supersede matching pending cache data. The declared size must be a nonnegative integer, chunk size positive; `read(n)` must return bytes no longer than n. Exactly the declared size is consumed; extra source bytes are not checked. Empty streams publish empty bytes without calling the reader.

`get_to_file` streams **raw stored bytes**, including any serialization header, not decoded Python objects. It handles short `write()` results by sending the unwritten suffix. A sink must return a positive integer no larger than the supplied buffer; `None`, zero, negative, oversized, and noninteger results fail clearly. An error may leave the external sink partially written. Neither streaming call automatically retries a stateful callback or a whole consumed stream.

File-backed streaming reads hold one snapshot across row lookup, length, and BLOB reads. A slow sink does not monopolize the writer connection, but can retain old WAL pages and delay checkpoint completion. Same-vault database access from a stream callback is rejected to avoid reentrant lock deadlocks; use callbacks dedicated to I/O, not nested vault operations. Memory/URI fallback streams still serialize on the primary connection.

When `size` is omitted, KohakuVault attempts to determine it via `seek/tell` or file descriptors; otherwise it buffers the input in memory before storing it. Use an explicit known size for bounded streaming.

## API Cheat Sheet

| Category | Methods |
|----------|---------|
| CRUD | `__getitem__`, `__setitem__`, `__delitem__`, `get`, `put`, `delete`, `exists`, `pop`, `setdefault`, `clear`, `__contains__`, `len` |
| Iteration | `keys(prefix=None, limit=None, batch_size=1024)`, `items(...)`, `values(...)` (unlimited by default) |
| Bulk | `put_many(items)`, `get_many(keys, default=None)` (explicit list/tuple batches), `update(other=(), **kwargs)` (bounded iterable batches) |
| Streaming | `put_file`, `get_to_file` |
| Caching | `enable_cache`, `disable_cache`, `flush_cache`, `cache(cap_bytes, flush_threshold, auto_flush=True)`, `lock_cache()` |
| Auto-pack | `enable_auto_pack`, `disable_auto_pack`, `auto_pack_enabled`, `enable_headers`, `disable_headers`, `headers_enabled` |
| Maintenance | `checkpoint`, `optimize`, `close`, `reader_pool_stats()`, `background_flush_error` (property) |

Missing indexed keys and `pop()` without a default raise `KeyError`; closed/invalid iteration use can raise `ValueError`. Other operation errors are mapped through `kohakuvault.errors`: `DatabaseBusy`, `NotFound`, `InvalidArgument`, `IoError`, or `KohakuVaultError` for unexpected failures.

## Interop Patterns

- **Share the DB with ColumnVault** – pass a `KVault` instance to `ColumnVault(kv)` to reuse its database file when mixing blobs and typed columns.
- **Metadata + Blobs** – keep large binary payloads in `KVault` while storing structured metadata in MessagePack columns.
- **Vector Pipelines** – auto-pack stores numpy arrays directly; pair this with `VectorKVault` for similarity search when you need indexing.

## Maintenance & Diagnostics

- Call `kv.checkpoint()` in long-running writers to keep the WAL small.
- `kv.optimize()` issues `PRAGMA optimize; VACUUM;`—handy after bulk deletes.
- Always `close()` or use context managers so pending data is flushed and native file handles are released. Do not rely on daemon-thread shutdown for durability.
- For 0.6.x compatibility, `disable_auto_pack()` returns you to the old bytes-only behaviour instantly.

## Testing & Benchmarks

- Pytests: `tests/test_basic.py`, `tests/test_cache_consistency.py`, `tests/test_storage_compatibility.py` cover CRUD, caching, retries, and streaming.
- Regression tests: `tests/test_kv_handoff.py`, `test_kv_flush_isolation.py`, `test_kv_batch.py`, and `test_kv_process_readers.py` cover streaming failures, blocked flushes, atomic batches, lifecycle, and four independent reader processes.
- Mapping/daemon/reentrancy coverage: `tests/test_kv_mapping.py`. Reader-policy benchmark: `python examples/benchmark_readers.py`.
- Benchmark: `python examples/benchmark_kv.py --structured` measures cached scalar acceptance/flush, committed native batches, point/bulk reads, iteration, and four shared-handle readers. See [measurements and regressions](performance.md). `examples/all_usage.py` exercises hybrid workflows.

Keep this guide handy when adding new methods—if it changes behaviour, document it here and link from `docs/README.md` so users know where to look.
