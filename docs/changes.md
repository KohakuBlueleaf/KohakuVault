# Storage and API changes since 0.9.0

## Compatibility and migration

**KV and vector databases keep their existing table layouts, headers and value encodings.** The standalone DataPacker format is unchanged. Regression tests read a database created by 0.9.0 and require byte-identical KV rewrites; the original extension also reads new KV/vector records alongside new column tables. This is targeted compatibility evidence, not a guarantee covering every historical version or malformed file.

**Columns have a new, incompatible physical format.** New `kvault_col_*` tables use an independent version marker. Legacy `col_*` data is neither read, migrated nor deleted. If you have legacy columns, export them with the old version and re-create/import them with this version before switching applications. Merely opening the old file does not make old columns appear in the new catalog.

WAL and `synchronous=NORMAL` remain unchanged. Buffered acceptance, committed visibility and power-loss durability are different guarantees. Do not copy a live main database file without its committed WAL state; use a consistent backup.

## New features

- Buffer-oriented columns: NumPy/tensor `append_array`, `read_array`, `read_buffer`, variable `read_buffers`, indexed `take`, and `iter_batches`. Reads fetch requested BLOB ranges rather than whole growing chunks.
- Atomic KV `put_many(list_or_tuple)` and ordered `get_many(keys, default=None)`. All entries in one explicit write batch commit together; duplicate keys use the last value. Numeric/structured values retain the existing serialization pipeline.
- Unlimited, keyset-paginated KV iteration with binary-prefix support and paired `items()` queries. `limit` caps total results; `batch_size` controls query size.
- File-backed independent readers, explicit `read_only=True`, snapshot-consistent multi-step reads, and pooled streaming output.
- Pending/in-flight KV cache ownership: committed and buffered reads proceed while SQLite blocks a cache flush. Failed flushes preserve newer pending replacements.
- KV reader policy controls `reader_cache_kb`, `reader_idle_limit`, inherited `mmap_size`, plus `reader_pool_stats()` for idle SQLite page-cache allocation.
- Transactional streaming overwrite, correct handling of short sink writes, retryable close, and nested cache-lock contexts.
- Vector input uses contiguous float32 buffers; vector/payload mutations commit atomically. Vector and column context managers release connections deterministically.

## Behavior changes requiring attention

| API | Change / action |
|---|---|
| Column storage | Incompatible with old column layout; export/import if needed. |
| Column slicing | Only unit-step slices; use `take()` for gathers. `read_array()` is read-only unless `copy=True`. Legacy chunk-tuning parameters are deprecated; use `block_bytes`. |
| KV iteration | No default 10,000-key cutoff. Supply `limit` when a bounded total is intended. |
| `update(other=(), **kwargs)` | Commits in batches of at most 1024 pairs instead of scalar cached writes. It bypasses cache buffering and rejects an enabled cache lock. Earlier batches survive a later failure; it is not whole-input atomic. Use a scalar loop for buffered updates or `put_many()` for a single explicit atomic batch. |
| `pop(key)` | Missing keys now raise `KeyError`; pass an explicit default (including `None`) for fallback. Removal is atomic and commits before decoding; decode errors do not undo deletion. |
| `setdefault(key, default=None)` | Supports typed defaults and optional None; returns decoded data rather than coercing defaults to bytes. Missing-key insertion commits directly and rejects an enabled cache lock. Existing defaults are not serialized. |
| `clear()` | One atomic SQL deletion, then pending retirement; SQL failure no longer leaves a partially cleared table. Commits directly like `delete()`. |
| Bytes-like KV values | `bytes`, `bytearray` and `memoryview` inputs are snapshotted as raw bytes consistently in scalar/bulk operations. Use an explicit serialization wrapper if preserving the Python container type is required. |
| `lock_cache()` | Nested scopes suppress future flush batches, not already in-flight commits. Not a transaction. Oversized bypass, close, or batch insert cannot silently discard locked pending data. |
| Background flushing | Failure warns, records `background_flush_error`, and stops that worker instead of silently swallowing errors. Retry with `flush_cache()`; successful unlocked flush clears the error. Explicitly re-enable a timed cache to restart the worker. |
| Streaming callbacks | `read(n)` must return bytes of at most n; sink `write()` must report positive integer progress, not None/bool. Zero/negative chunk sizes fail. Failed whole streams are not automatically replayed; same-vault stream callback reentry is rejected instead of deadlocking. |
| Scalar retry configuration | `retries` must be positive, backoff finite/nonnegative, and background intervals finite/positive. |
| Vector search metric | A query metric differing from the stored index metric now raises rather than being silently ignored. |
| Close/lifetime | Closed handles reject operations. Failed final flush leaves the handle retryable; close inside a KV cache-lock context is rejected. KV close now releases its writer/idle readers even if a traceback retains the native wrapper; in-progress readers retain their lease only until return. |

Serializer/deserializer callbacks run outside packer/database/cache locks and may reenter KV/vector APIs. Their nested side effects are independent operations, not rolled back with an outer serialization failure. Stream callbacks are more restricted because stream writes own a database transaction.

## Scope and operational limits

SQLite still serializes writers; no server, custom pager, ANN replacement, compression or public cross-column transaction API is added. Memory/SQLite-URI KV readers use the primary connection. Idle retention is not an active-reader cap; each active reader can allocate its own cache. Slow stream sinks can retain WAL snapshots. Iterators use fresh snapshots per page/batch, not one immutable whole-iteration snapshot. Explicit bulk calls materialize caller-bounded input. Column structural/variable-width updates remain replacement operations.

The native collection suite now compiles and runs. SkipList insert-if-absent behavior is unchanged; its stale Rust test is aligned with the existing Python contract, not changed to replacement semantics.

See [KV guide](kvault.md), [column guide](columnvault.md), [vector guide](vectors.md), and [measured performance](performance.md) for details and runnable examples.
