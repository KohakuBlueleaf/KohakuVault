# Column storage design

## Goals and compatibility boundary

Keep SQLite's durable storage, WAL, and transaction implementation. Replace the unused column format and the execution paths that repeatedly materialized large chunks or converted contiguous batches back into per-element objects.

The format version in `kvault_col_format` is independent of KV metadata. KVault tables, auto-pack headers, standalone DataPacker serialization, sqlite-vec tables, and vector payload encodings are unchanged. Old column tables are not migrated or reused. Opening a database with an unknown new-column format version fails explicitly.

## Module map

| Module | Responsibility |
|---|---|
| `src/kvault-rust/col/mod.rs` | Thin PyO3 bindings, error translation, GIL boundaries |
| `col/core.rs` | Catalog, format initialization, connection lifetime, transaction boundaries |
| `col/chunks.rs` | Directory entries, bounded allocation, incremental BLOB helpers |
| `col/fixed.rs` | Fixed-width append, partial reads/updates, indexed gathers |
| `col/varsize_ops.rs` | Variable payload/offset append and coalesced range reads |
| `common/read_pool.rs` | Independent short-lived read snapshots and bounded idle connection reuse |
| `src/kohakuvault/column_codec.py` | Numeric/tensor buffers and generic object codecs |
| `column_proxy.py` | Public API, contiguous append caches, lifecycle and iteration |

No alternate `col2` implementation or legacy column compatibility layer is retained.

## Physical format

`kvault_col_columns` identifies each logical column: unique name, dtype, element width, committed row count, and block byte target. Column IDs are monotonic so a stale handle cannot accidentally address a newly created column with a recycled ID.

`kvault_col_blocks` contains only small mutable directory records:

```
block_id, col_id, block_no, first_row, row_count,
used_bytes, capacity_bytes, row_capacity
```

Its indexes are `(col_id, block_no)` and `(col_id, first_row)`. `kvault_col_data` contains the payload BLOB and optional offsets BLOB, addressed by the same block ID.

**Directory metadata and payloads are separate SQLite rows.** Updating a row count must not reconstruct the row containing a large BLOB. BLOB handles are opened only within a read or write transaction and are closed before publication/commit.

### Fixed-width blocks

Rows per block are `max(1, block_bytes // elem_size)`. Row index arithmetic determines the block and byte offset directly. No scan or sum over earlier block sizes is required.

A block is preallocated with `zeroblob(rows_per_block * elem_size)`. Appends write the new range with incremental BLOB I/O. Existing bytes are not fetched or copied to expand a tail. The final partially filled block may retain unused capacity.

Primitive and fixed-shape tensor columns contain little-endian scalar values. Shape and dtype are stored once in the catalog. Generic fixed-width encodings such as padded MessagePack use the unchanged DataPacker representation.

### Variable-width blocks

Each block has a bounded payload capacity and a separate u64 offsets buffer. For N committed rows, N+1 offsets delimit their byte ranges. Empty records repeat an offset.

An ordinary block's row capacity is `clamp(block_bytes // 16, 1, 65536)`. This bounds offset storage even for empty/tiny records. Either payload capacity or offset capacity can seal a block. Records larger than the target block size receive a dedicated block; no record may exceed 512 MiB.

An append writes the largest consecutive input span that fits in the current tail, writes its offsets once, updates small directory metadata, and continues in another block when necessary. It never concatenates new bytes onto an ever-growing persistent BLOB.

## Write path

1. Validate and serialize the Python input. Numeric batches become one contiguous little-endian buffer; variable batches become payload plus offsets.
2. Acquire the native writer connection and begin an immediate transaction before reading the current append position.
3. Locate or allocate bounded blocks and write their BLOB ranges.
4. Update directory entries and the committed column length.
5. Commit, then return success.

A direct batch is atomic even when it spans multiple blocks. Variable payload and offsets cannot be published independently. Injected SQLite failures roll back the complete batch. SQLite still serializes write transactions; there is no new parallel-writer algorithm.

Python-side append caches hold contiguous bytearrays, not an object per numeric element. Capacity includes offset entries for variable records. Buffer contents are cleared only after a successful native commit. A close operation stops new buffered admissions before flushing, preventing a concurrently serialized append from being acknowledged after close.

## Read path

A file-backed read leases an independent read-only SQLite connection, begins a short transaction, reads metadata and payload in the same snapshot, and rolls back the read transaction before returning the connection to the idle pool. Up to eight idle connections are retained. Active reads are not serialized by the pool mutex. In-memory databases use the primary connection instead.

A fixed range allocates the requested output size and reads only the corresponding BLOB ranges. A variable range fetches the necessary offsets, then one contiguous payload range per affected block. The low-level variable API returns flat payload and offsets buffers.

Fixed gathers sort row references by block and scatter results back to the caller's requested order. They currently issue individual element reads inside each opened block; they do not implement a general read-coalescing optimizer.

SQLite I/O runs without the Python GIL. Numeric arrays are views over returned owned Python bytes and do not pin a SQLite snapshot. The current binding still has native-to-Python output copying; claiming end-to-end zero-copy would be incorrect.

## Concurrency and lifecycle

Normal opens of an initialized column database do not rerun schema creation. Read-only opens perform no catalog writes or journal-mode changes. Writer reservation precedes append-position reads, preventing stale offsets between independent writers.

Each read call is snapshot-consistent. A long Python iterator uses short per-batch snapshots and an initial row-count high-water mark, avoiding an indefinitely pinned WAL reader. That is suitable for append-only data, not a guarantee of whole-iteration consistency under arbitrary concurrent mutation.

`close()` waits for native operations and releases primary and pooled connections. Python column handles share a small lifetime state rather than holding a strong reference cycle to their parent vault. Context managers provide deterministic flush/close behavior on Windows as well as other platforms.

## Explicit non-goals and remaining costs

There is no query planner, compression, predicate pushdown, null bitmap, ANN replacement, or multi-column transaction API. Structural mutation and variable-width replacement currently rebuild the complete column through a slow convenience path. Variable-width iteration uses a row-size estimate rather than a hard byte ceiling. Those restrictions are documented in the public guide.

The new engine retains WAL with `synchronous=NORMAL`, matching the previous performance-oriented default. Transaction consistency is not a promise that every acknowledged commit survives sudden power loss. Full power-failure durability requires a separately exposed/configured synchronization policy; this rewrite does not silently weaken or disable journaling to produce benchmark gains.

## Validation

The Python suite covers old API behavior and new buffer paths. Added regressions exercise cross-block reads, endian/noncontiguous arrays, array ownership, gather order, empty/oversized variable records, bounded caches, retryable flush failures, stale handles, read-only opens under a writer lock, threads, and an independent-process reader.

The compatibility fixture was created by the original 0.9.0 extension before replacement. Tests verify old KV/vector reads, byte-identical KV rewrites, and new-column coexistence. See `tests/fixtures/README.md` for provenance and `tests/test_storage_compatibility.py` for assertions.
