# KohakuVault

SQLite-backed storage with a Rust engine, Python APIs, and batteries included for blobs, typed columns, and cache-aware indexes.

## Quick Start

```bash
pip install kohakuvault
```

```python
from pathlib import Path

from kohakuvault import ColumnVault, DataPacker, KVault

db_path = Path("kohaku.db")

# Key-value blobs
kv = KVault(db_path)
kv["cover:001"] = b"\x89PNG..."  # store large blobs without buffering
assert kv.get("cover:001").startswith(b"\x89PNG")

# Columnar data shares the same SQLite file
cols = ColumnVault(kv)
temps = cols.ensure("temperatures", "f64")
temps.extend([23.6, 23.8, 24.1])

profiles = cols.ensure("profiles", "msgpack")
profiles.append({"id": 1, "name": "Rin", "active": True})
assert profiles[0]["name"] == "Rin"

# DataPacker for standalone serialization
packer = DataPacker("bytes:32")
token = packer.pack(b"session")
assert packer.unpack(token, 0).rstrip(b"\x00") == b"session"
```

## Storage Interfaces at a Glance

| Interface            | Data model              | Access pattern        | Backing tables / structures            | Highlights                                         | Best for                               |
|----------------------|-------------------------|-----------------------|----------------------------------------|----------------------------------------------------|----------------------------------------|
| `KVault`             | Key -> opaque bytes      | Dict-style            | `kvault` (blob)                        | WAL-friendly streaming, retry-aware operations     | Blobs, media, large files              |
| `Column`             | Fixed-size elements     | Mutable sequence      | `col_meta` + `col_chunks`              | Batch slice read/write, Rust packing fallback      | Numeric telemetry, dense metrics       |
| `VarSizeColumn`      | Prefixed variable bytes | Mutable sequence      | `{name}_data` + `{name}_idx`           | Size-aware updates, adaptive chunk growth          | Logs, JSON payloads, text              |
| `DataPacker`         | Typed serializer        | Pack/unpack helpers   | Pure Rust (no extra tables)            | MessagePack/CBOR, fixed/variable strings & bytes   | Preprocessing, custom pipelines        |
| `CSBTree`            | Ordered map             | B+Tree style API      | Arena-backed cache-sensitive tree      | Contiguous nodes, iterator & range queries         | Sorted secondary indexes, metadata     |
| `SkipList`           | Ordered map             | Lock-free (CAS)       | Lock-free skip list                    | Concurrent inserts/reads without GIL contention    | Shared read/write heaps, hot paths     |

## Capabilities

- Rust-powered I/O with Python-first ergonomics (PyO3 bridge).
- Write-back cache for both key-value and columnar workloads (context manager, daemon auto-flush, capacity guards).
- Automatic dtype parsing with DataPacker fallback for legacy pack/unpack helpers.
- Fast range access: `Column.__getitem__` batches reads; slice assignment funnels to Rust.
- Variable-size column maintenance: prefix-sum index, chunk rebuilds, and fragment tracking.
- Concurrency aware retry logic (`_with_retries`) that turns SQLite busy states into typed exceptions.
- Optional CSB+Tree and SkipList implementations for ordered access patterns in the same extension module.

## Architecture

```
Python layer (proxy.py / column_proxy.py)
    -> PyO3 bindings
Rust core (lib.rs)
    -> rusqlite + custom allocators
SQLite storage (single .db + WAL)
```

- **KVault**: mutex-protected connection, optional write-back cache, streaming via BLOB API.
- **ColumnVault**: element-aligned chunking, cache buckets per column, adaptive variable-size slices.
- **DataPacker**: Rust serializers report `elem_size` / `is_varsize` to Python, enabling automatic dtype strategy.
- **SkipList / CSBTree**: share Python key wrappers to support arbitrary `PyObject` ordering.

## Performance Snapshot (M1 Max, 50K entries)

- KVault write with 64 MiB cache: ~24k ops/s at 16 KiB payloads (~377 MB/s).
- KVault read hot cache: ~63k ops/s at 16 KiB payloads (~987 MB/s).
- Column `extend` (`i64`): ~12.5M ops/s with cache (~95 MB/s), >450x faster than uncached append loops.
- Column slice read (`f64`, 100 items): ~2.3M slices/s, 200x faster than per-element fetch.
- MessagePack column writes: >1M pack/unpack ops/s with Rust DataPacker.

See `examples/benchmark.py` and `examples/benchmark_container.py` for reproducible scripts.

## Tooling & Extras

- **DataPacker**: supports primitives, strings (`utf8`, `utf16le`, `latin1`, `ascii`), `bytes:N`, `msgpack`, `cbor`, and JSON Schema validation. `pack_many`/`unpack_many` enable zero-Python loops for batch work.
- **Write-back cache**: `vault.cache(...)` and `column.cache(...)` manage flush thresholds, auto daemon threads, and locking (`lock_cache`) to coordinate multi-column writes.
- **Mixed workloads**: share a single SQLite file between KVault and ColumnVault (pass a `KVault` instance into `ColumnVault`).
- **Ordered indexes**: `CSBTree` (cache-sensitive B+Tree) and `SkipList` shipped in `_kvault`; import from `kohakuvault` when the extension is built.
- **Error mapping**: raw `RuntimeError` from Rust is translated into `KohakuVaultError`, `DatabaseBusy`, `NotFound`, `InvalidArgument`, or `IoError` via `errors.map_exception`.

## Development & Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Format and lint
ruff check --fix .
black src tests examples
cargo fmt

# Run Python + Rust tests
pytest
cargo test
```

The repository uses maturin for building the extension; see `pyproject.toml` for configuration.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
