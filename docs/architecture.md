# KohakuVault Architecture

KohakuVault is a pure-Python package backed by a Rust extension that talks to SQLite. Python keeps the ergonomic APIs (`Mapping`, `MutableSequence`, context managers, retries) while Rust handles serialization, chunk orchestration, and tight loops. Everything eventually lands in a single SQLite database so deployment stays trivial.

## Layered Model

| Layer | Location | Responsibilities |
|-------|----------|------------------|
| Python proxies | `src/kohakuvault/proxy.py`, `column_proxy.py`, `vector_proxy.py`, `wrappers.py` | Public API, argument validation, retry/backoff, cache orchestration, type wrappers for auto-pack |
| PyO3 module | `src/kvault-rust/lib.rs` | Exposes `_KVault`, `_ColumnVault`, `DataPacker`, CSBTree, SkipList, and `_VectorKVault` to Python |
| Rust subsystems | `src/kvault-rust/kv`, `col`, `packer`, `tree`, `skiplist`, `vkvault`, `vector_utils.rs` | SQLite sessions (rusqlite), chunk/cache management, serialization, ordered containers, sqlite-vec integration |
| Storage | SQLite database (`.db`, `.db-wal`, `.db-shm`) | ACID persistence, WAL concurrency, single deployment artifact |

## Data Layout

All components share one SQLite database file. Default table names are configurable but the on-disk patterns stay consistent:

- **KVault** – `kv` (default) with `key BLOB PRIMARY KEY, value BLOB NOT NULL`. Values may contain a 10-byte auto-pack header (see below) or raw bytes.
- **Column metadata** – `col_meta` stores column id, dtype string, logical length, min/max chunk sizes, and cache hints.
- **Column chunks** – `col_chunks` persists payload blobs keyed by column id and chunk id. Fixed-size columns append contiguous elements, while variable-size columns keep chunk usage metadata.
- **Variable-size indexes** – for each var-size column we maintain `{name}_data` + `{name}_idx`. Index rows hold 12-byte triples `(chunk_id, start, end)` so we can jump straight to the byte range.
- **Vector search** – `_VectorKVault` creates a sqlite-vec `vec0` virtual table for dense vectors plus a companion blob table for arbitrary values. Rows reference the value blob id so vector scans remain cache-friendly.
- **Metadata** – `kohakuvault_meta` records schema versions for migrations and sanity checks.

WAL (Write-Ahead Logging) keeps readers isolated from writers. Both `_KVault` and `_ColumnVault` expose `checkpoint()` helpers so long-running jobs can merge WAL pages back to the main file.

## Auto-Pack Pipeline

Auto-packing lets `KVault` accept rich Python objects without manual serialization. The Rust implementation in `kv/autopacker.rs` runs through a priority order:

1. `bytes` → stored as-is (no header, zero overhead).
2. Explicit wrappers (`MsgPack`, `Json`, `Cbor`, `Pickle` from `kohakuvault.wrappers`) → force an encoding when you need one.
3. NumPy arrays → `vec:*` DataPacker dtype.
4. Int/float → `i64`/`f64` DataPacker dtypes.
5. `str` → UTF-8 bytes.
6. dict/list → MessagePack with a Pickle fallback.
7. Everything else → Pickle if enabled.

Non-raw values are prefixed with a 10-byte header defined in `kv/header.rs`:

```
|0x89 0x4B|version|encoding|flags|reserved(3)|0x56 0x4B|
```

`encoding` tracks MessagePack/CBOR/DataPacker/etc. The header guard makes it safe to mix raw blobs (images, audio) with auto-packed objects in the same table. `KVault.enable_auto_pack()` and `disable_auto_pack()` toggle the behaviour, and `auto_pack_enabled()` reports the current state. Wrappers simply give callers an opt-in way to override the default encoding choice (for example, force JSON bytes for downstream tools).

## Connections, Caches, and Concurrency

- Each `_KVault` holds a `rusqlite::Connection` behind a `Mutex`. Write-back caches live in Rust (`WriteBackCache`) and are flushed via explicit calls (`flush_cache`, context managers) or daemon timers started from Python.
- `_ColumnVault` owns its own connection plus per-column caches stored in a `Mutex<HashMap<ColumnCache>>`. The caches buffer appends/updates until thresholds are hit or manual flushes occur.
- Python’s `_with_retries` helper wraps every call, converting rusqlite errors into typed Python exceptions (`DatabaseBusy`, `NotFound`, `InvalidArgument`). Busy loops use exponential backoff before surfacing an error.
- Cache lock contexts (`kv.lock_cache()`, `column.cache().lock_cache()`) temporarily block auto-flushes so multi-step transactions can batch writes safely.

## Chunking Strategy

Fixed-size columns align chunk boundaries to `elem_size`, so slicing `col[1_000:2_000]` can stream contiguous blobs without reassembling boundaries. Variable-size columns append to `_data` chunks while `_idx` keeps offsets. When an update outgrows its original chunk, the Rust backend rebuilds the affected chunk and updates metadata atomically.

Chunk growth is exponential between `min_chunk_bytes` and `max_chunk_bytes` (defaults 128 KiB → 16 MiB). You can override these when creating columns to match your workload (e.g., smaller chunks for high-churn logs, larger chunks for wide analytics tables).

## Vector Search Stack

`VectorKVault` builds on sqlite-vec 0.1.6:

- `VectorKVault.insert()` normalizes numpy arrays/lists/bytes into float lists and forwards them to the Rust `_VectorKVault` which inserts into the `vec0` virtual table plus a blob table for values.
- Queries (`search`, `get`) run brute-force scans accelerated by SIMD (AVX/NEON) inside sqlite-vec. Metrics include `cosine`, `l2`, `l1`, and `hamming`. Per-query overrides are supported.
- Maintenance APIs (`get_by_id`, `update`, `delete`, `exists`, `count`, `info`) expose the sqlite-vec rowid, making it easy to track metadata in other tables.

Because sqlite-vec currently performs full scans, WAL checkpoints and VACUUMs are important for keeping the file compact and query times predictable.

## Maintenance Hooks

| Operation | Python method | Notes |
|-----------|---------------|-------|
| WAL checkpoint | `KVault.checkpoint()`, `ColumnVault.checkpoint()` | Calls into `common::checkpoint_wal` |
| Optimization | `KVault.optimize()` | Runs `PRAGMA optimize; VACUUM;` |
| Cache flush | `flush_cache()` on vaults/columns | Forces buffered writes into SQLite |
| Auto-pack headers | `KVault.enable_headers()`/`disable_headers()` | Control whether raw bytes get headers (rarely needed) |

## Repository Map

```
KohakuVault/
├── src/
│   ├── kohakuvault/
│   │   ├── proxy.py          # KVault proxy & auto-pack controls
│   │   ├── column_proxy.py   # ColumnVault proxies & dtype parsing
│   │   ├── vector_proxy.py   # VectorKVault proxy
│   │   ├── wrappers.py       # MsgPack/Json/Cbor/Pickle wrappers
│   │   └── errors.py         # Exception hierarchy
│   └── kvault-rust/
│       ├── kv/               # Key-value subsystem & auto-pack header
│       ├── col/              # Column subsystem & caches
│       ├── packer/           # DataPacker
│       ├── vkvault/          # Vector search
│       ├── tree/             # CSBTree
│       └── skiplist/         # Lock-free skip list
├── docs/                     # Guides (this folder)
├── examples/                 # Benchmarks and demos
└── tests/                    # Pytest suite covering KV/columns/vectors
```

Keep this layout handy when onboarding new contributors—most bugs fall into “Python proxy”, “Rust backend”, or “SQLite plumbing”, and this map shows where to dive in.
