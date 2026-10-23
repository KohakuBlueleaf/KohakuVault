# KohakuVault Architecture

KohakuVault combines Python APIs, a Rust/PyO3 extension, and SQLite persistence. It requires no database service. All durable collections live in one SQLite database; WAL mode can create `-wal` and `-shm` sidecars while the database is in use. Copying only the main file during active writes is not a safe backup procedure.

## Layers

| Layer | Location | Responsibility |
|---|---|---|
| Public APIs | `src/kohakuvault/*_proxy.py`, `proxy.py` | Mapping/column/search interfaces, validation, batching, lifecycle |
| Column codecs | `column_codec.py` | Contiguous numeric/tensor buffers and generic record conversion |
| Native storage | `src/kvault-rust/kv/`, `col/`, `vkvault/`, `textvault/` | Transactions, BLOB I/O, SQL queries, native serialization |
| Shared infrastructure | `common/`, `common/read_pool.rs` | Connection setup and read snapshots |
| KV buffering/batches | `kv/cache.rs`, `kv/batch.rs` | Pending/in-flight ownership, explicit native bulk calls |
| Durable substrate | SQLite and sqlite-vec | Persistence, locking, recovery, vector-table representation |

## Storage and compatibility

KV uses its existing `key BLOB PRIMARY KEY, value BLOB NOT NULL` tables. Raw bytes remain raw. Automatically packed values retain the existing ten-byte header and MessagePack/CBOR/DataPacker/Pickle encodings. `kv/header.rs`, `kv/autopacker.rs`, and `packer/` remain the compatibility boundary; column optimizations do not change them.

Vector storage retains the sqlite-vec 0.1.6 `vec0` table and its companion `{table}_values` table. Insert, update, and delete now publish the vector and associated payload atomically. Normalizing and packing float vectors is unchanged on disk; the Python input path passes contiguous float32 bytes instead of a Python list of coordinates.

Columns use a replacement format with an independent version marker:

| Table | Content |
|---|---|
| `kvault_col_format` | Column format version |
| `kvault_col_columns` | Name, dtype, fixed width, length, block target |
| `kvault_col_blocks` | Small mutable block-directory records |
| `kvault_col_data` | Payload BLOBs and optional u64-offset BLOBs |

Mutable directory metadata is deliberately separate from payload rows. Fixed blocks are addressed arithmetically; variable blocks use a row-position index. Both use bounded preallocation rather than repeated persistent BLOB growth. The old `col_*` format is not migrated, read, or deleted. See [column storage design](columnar_storage.md) for details.

TextVault retains its FTS5 tables and existing value encodings. Standalone CSBTree/SkipList containers remain independent in-process utilities, not persistent column indexes.

## Connections and concurrency

File-backed KV point reads/scans, column reads, and vector queries use independent read-only connections. By default each pool retains at most eight idle connections; KV exposes `reader_idle_limit` and `reader_cache_kb` (8 MiB per reader by default), and propagates `mmap_size`. Idle retention does not cap active readers; the pool lock is not held while executing a query. Single-statement KV point queries use SQLite's implicit statement snapshot; explicit read transactions cover multi-step metadata/payload and bulk accesses. `reader_pool_stats()` reports idle SQLite cache allocations, excluding mmap, active leases and Python memory. In-memory stores retain a single connection and serialize access.

`read_only=True` is available for KVault, ColumnVault, and VectorKVault. It opens an existing database without schema creation or journal-mode changes. SQLite work on the new read/write paths releases the GIL. Python result decoding occurs after a read snapshot is released. sqlite-vec extension registration is guarded by a process-wide `Once`.

Writes remain serialized by SQLite. Column appends and vector compound mutations use an immediate transaction, acquiring the writer reservation before reading mutable positions/references. A separate writer connection allows existing readers to proceed under WAL. There is no external daemon, distributed writer, cross-interface transaction API, or custom recovery mechanism.

KV stream writes allocate, copy, validate and commit in one immediate transaction. File-backed stream reads use a pooled connection and one snapshot; slow sinks can retain WAL pages. Callback reentry into the same vault is rejected. Known-size streams do not replay consumed callbacks on errors. Sharing a file across vault interfaces does not imply sharing a transaction or an in-memory SQLite connection.

## Caches and iteration

KV caches use an ordered pending map and one immutable in-flight batch with shared payload ownership. Read precedence is pending → in-flight → committed SQLite. Capacity counts both batches' key/value bytes. Scans take bounded ranges of references from each map, not a copy of all pending payloads for every page.

**Lock order and publication:** cached admission takes a shared mutation gate then briefly locks cache state. Direct, streamed, batch, delete and lifecycle mutations take the exclusive mutation gate, then the writer connection, then short cache-state locks. Flush takes the writer connection, moves pending into in-flight under the cache lock, releases that lock for all SQL/commit work, and retires or recovers the batch before releasing the connection. Flush does not take the mutation gate, allowing newer cached admissions while it waits. Readers never take the mutation gate or file-backed writer connection. No cache-state mutex is held across SQLite or Python callbacks. Uncontended memory-only admissions use try-locks under the GIL; contended paths detach before waiting. Serialization/deserialization captures a copy of the packer policy and releases its mutex before Python callbacks, including on vector paths.

Successful flush retires only its batch. Failed flush restores unsuperseded entries; newer pending values win. The writer connection serializes recovery before later direct writes can commit. Direct/stream/batch writes and deletes remove superseded pending entries only after successful SQL, while still owning the writer connection and exclusive admission gate. Deletes are synchronous committed operations, not buffered tombstones: by acknowledgment, older in-flight writes have resolved and pending data is removed, so there is no deferred delete overlay to resurrect. Failure leaves the old pending value recoverable.

`lock_cache()` suppresses future batch publication, not an already in-flight transaction. Nested Python contexts count active locks. Reset/disable/close use exclusive admission and a required flush, so skipped locked batches cannot be mistaken for successful final publication. Native close marks the handle closed only after success; writers recheck admission after serialization. In-progress read calls retain native ownership until return. Background flush failure records an inspectable error, warns, and stops the worker; acknowledged pending data remains recoverable by explicit flush.

`put_many()` serializes an explicit list/tuple before taking database locks and commits all pairs in one transaction (last duplicate wins). `get_many()` captures matching cache references and runs all committed lookups in one snapshot, then decodes outside the database locks. This is not one combined cache/database snapshot under concurrent mutation. `update()` delegates bounded groups of 1024 pairs to the committed batch path. `clear()` uses one atomic DELETE; `pop()` and `setdefault()` combine lookup/mutation under an immediate transaction. These calls fence pending admission and preserve pending state on SQL failure.

Column append caches are contiguous Python bytearrays. Numeric batches avoid splitting a buffer into per-element Rust/Python allocations. Variable caches add a contiguous offsets buffer. Successful flushes publish one transaction; failed flushes retain pending data. A cache context flushes on exit and is not a rollback transaction.

KV `keys()`, `items()`, `values()`, and ordinary iteration are unlimited by default. `limit` is an optional total cap; `batch_size` controls each query. Keyset pagination replaces a one-shot limited scan. `items()` reads key/value pairs together rather than issuing an extra lookup per key. Binary prefixes use an increment-and-truncate upper bound, including empty and all-0xff prefixes.

Pages use fresh snapshots, not one pinned iterator-wide snapshot. Column iterators likewise capture an initial end row and read batches separately. Concurrent destructive changes do not provide immutable whole-iteration semantics.

## Lifetime and durability

Use context managers or explicit `close()`. Column closure flushes pending data, waits for native operations, and closes pooled/primary connections. KV/vector closure releases their native handles. Avoid retaining unused writer handles or long-lived read transactions.

WAL with `synchronous=NORMAL` remains the performance-oriented default. Atomic commit is not a guarantee that every acknowledged transaction survives sudden power loss. Journaling has not been disabled to obtain benchmark results. Column `checkpoint()` reports `(busy, log_frames, checkpointed_frames)`; committing and checkpointing are distinct.

## Validation and measurements

The compatibility suite uses an actual database created by the old extension and checks byte-identical KV rewrites alongside legacy vector reads. New column tests cover buffers, cross-block ranges, rollback, empty/oversized records, read-only handles, threaded readers/writers, and a separate-process reader.

The basic benchmarks are `examples/benchmark_columnar.py` and `examples/benchmark_kv.py`. They distinguish buffer/batch throughput from scalar Python conversion and report elapsed time as well as useful payload MiB/s. See [performance notes](performance.md); measurements are workload-specific, not a general claim about SQLite's maximum throughput.
