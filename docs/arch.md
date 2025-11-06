# KohakuVault Architecture

KohakuVault binds a pure-Python API to a Rust core that persists to a single SQLite database. This document explains how the pieces cooperate, why specific choices were made, and what to keep in mind when extending the system.

## Layered Model

| Layer | File(s) | Responsibilities |
|-------|---------|------------------|
| Python proxies | `src/kohakuvault/proxy.py`, `src/kohakuvault/column_proxy.py` | User-facing interfaces, argument validation, retry/backoff policy, cache orchestration |
| PyO3 module | `src/kvault-rust/lib.rs` | Exports `_KVault`, `_ColumnVault`, `DataPacker`, `CSBTree`, `SkipList` to Python |
| Rust subsystems | `src/kvault-rust/kv`, `src/kvault-rust/col`, `src/kvault-rust/packer`, `src/kvault-rust/tree`, `src/kvault-rust/skiplist` | SQLite access (rusqlite), chunk/cache management, serialization, ordered containers |
| Storage | SQLite database (`.db`, `.db-wal`, `.db-shm`) | ACID persistence, WAL-based concurrency |

The Python layer retains ergonomic control (context managers, `Mapping` protocol) while the Rust layer handles performance-sensitive work (BLOB streaming, chunk layout, serialization).

## Data Layout

All components share a **single SQLite file**:

- **KVault table** (`kvault` by default) stores `key BLOB PRIMARY KEY, value BLOB`.
- **Column metadata** (`col_meta`) tracks column id, dtype string, element size, logical length, chunk configuration.
- **Column chunks** (`col_chunks`) persist payload bytes; fixed-size columns pack consecutive elements, variable-size columns store chunk fragments with usage metadata.
- **Variable-size indexes** use paired columns `{name}_data` + `{name}_idx` (12-byte triples: chunk id, start byte, end byte).
- **Metadata table** (`kohakuvault_meta`) records schema versioning for migrations.

SQLite Write-Ahead Logging keeps readers isolated from writers. `_KVault.checkpoint_wal()` and `_ColumnVault.checkpoint()` expose manual checkpoints so long-running processes can merge WAL changes.

## Execution Model

### Connections & Synchronisation

- Each `_KVault` owns a `rusqlite::Connection` guarded by `Mutex<Connection>`. Cache state (`WriteBackCache`) is also mutex-protected.
- `_ColumnVault` maintains a `Mutex<Connection>` plus per-column cache buckets (`Mutex<HashMap<i64, ColumnCache>>`) and an `AtomicBool` to coordinate cache locking.
- Python retries busy errors via `_with_retries` (exponential backoff). Rust functions bubble up `rusqlite::Error` which are mapped to `DatabaseBusy`, `NotFound`, or `KohakuVaultError`.

### Chunking Strategy

- **Fixed-size columns** align chunk sizes to element sizes to guarantee element boundaries. Chunk growth is exponential between `min_chunk_bytes` and `max_chunk_bytes`.
- **Variable-size columns** append to `{name}_data` while writing prefix sums to `{name}_idx`. Adaptive chunk rebuilds occur when inserts exceed chunk capacity.
- **Caches** store append buffers per column; when thresholds are reached, data is flushed through `_ColumnVault` or `_KVault` in a single transaction.

### Serialization Path

- `Column` and `VarSizeColumn` attempt to instantiate a Rust `DataPacker`. If unavailable, the Python fallback (`DTYPE_INFO`) handles primitive packing/unpacking.
- `parse_dtype()` uses `DataPacker` metadata (`elem_size`, `is_varsize`) to decide chunk configuration, keeping legacy dtype parsing for unsupported formats.

## Python <-> Rust Boundaries

| Python call | Rust entry point | Notes |
|-------------|-----------------|-------|
| `KVault.put`, `KVault.get` | `kv::ops::*` | Pack/unpack bytes, optional cache interception |
| `KVault.put_file`, `get_to_file` | `kv::stream::*` | Uses SQLite incremental BLOB API |
| `Column.append/extend` | `col::fixed` or `col::varsize_ops` | Accepts typed values or raw bytes depending on dtype |
| `Column.__getitem__` (slice) | `col::fixed::read_range_impl` / `col::varsize_ops::batch_read_varsize[_unpacked]` | Rust collects into contiguous buffers before returning to Python |
| `VarSizeColumn.__setitem__` | `col::varsize_ops::update_varsize_element` | Handles chunk rebuilds and fragment bookkeeping |
| `ColumnVault.enable_cache` | `ColumnCache` | Tracks buffers, idle flush timers, daemon thread coordination |

PyO3 handles `Py<PyAny>` lifetimes; tree/skiplist modules wrap Python objects via `PyObjectKey` and `PyValue` which manually manage reference counts with `Py_INCREF`.

## Error Handling

- Rust raises `PyRuntimeError` with a canonical message; Python proxies convert them via `errors.map_exception`.
- Cache operations surface `CacheError::{ValueTooLarge, NeedFlush}` which translate into controlled flush behaviour.
- IO errors when streaming are wrapped into `IoError` with the original exception as `__cause__`.

## Optional Structures

### DataPacker

- Located in `src/kvault-rust/packer`. Supports primitives (`i64`, `f64`), strings with multiple encodings, raw bytes, MessagePack, CBOR, and JSON Schema validation (feature flag `schema-validation`).
- Exposed to Python for standalone use and automatically consumed by `Column`/`VarSizeColumn`.

### CSBTree & SkipList

- `tree` module implements a cache-sensitive B+Tree with contiguous arena storage; exported as `CSBTree` and configured through a factory supporting `PyObject`, numeric, and text keys.
- `skiplist` module offers a lock-free skip list with atomic CAS operations; Python bindings wrap values in `Arc` for cheap clones across threads.

## Lifecycle & Maintenance

- **Checkpointing**: `KVault.checkpoint()` and `ColumnVault.checkpoint()` invoke `common::checkpoint_wal`.
- **Optimisation**: `_KVault.optimize()` triggers `PRAGMA optimize; VACUUM;`.
- **Cache lifespan**: Cache daemons run under Python thread management. They check idle durations using Rust-provided timestamps (`get_cache_idle_time`).
- **Resource cleanup**: `ColumnVault.__del__` ensures WAL checkpointing to avoid lingering journal files when the object is GC'd.

## Design Trade-offs

1. **Single SQLite file** keeps deployment trivial, but concurrent writers are limited by WAL semantics (single writer).
2. **Rust-first serialization** avoids Python loops, at the cost of optional compilation requirements (extension module must be built).
3. **Chunk-based columns** favour append-heavy workloads; random insert/delete requires chunk rewrites.
4. **Write-back caches** trade durability for throughput; callers must flush or rely on automatic guards.
5. **Ordered structures in extension** provide Python APIs without additional dependencies, but require the compiled module.

## File Map

```
KohakuVault/
|-- src/
|   |-- kohakuvault/          # Python package
|   |   |-- proxy.py          # KVault proxy
|   |   |-- column_proxy.py   # ColumnVault proxies
|   |   `-- errors.py         # Exception hierarchy
|   `-- kvault-rust/          # Rust crate
|       |-- kv/               # Key-value subsystem
|       |-- col/              # Column subsystem
|       |-- packer/           # DataPacker
|       |-- tree/             # CSBTree
|       `-- skiplist/         # Lock-free skip list
|-- docs/                     # Usage & architecture guides
|-- examples/                 # Benchmark and demo scripts
`-- tests/                    # Pytest suite covering KV, columns, tree/skiplist
```

Understanding the above layout helps new contributors navigate the codebase and reason about timing-sensitive paths (rust) versus high-level orchestration (python).
