# ColumnVault: buffer-oriented column storage

ColumnVault stores typed arrays and variable-width records in the same SQLite file as KVault and VectorKVault. The replacement engine prioritizes batch append, contiguous reads, small range reads, and independent readers.

**Compatibility:** KV/vector tables and value encodings are unchanged. Columns use the new `kvault_col_*` schema. Legacy `col_*` tables are left untouched but are not read or migrated. The standalone DataPacker format is unchanged.

## Numeric and tensor batches

```python
import numpy as np
from kohakuvault import ColumnVault

with ColumnVault("dataset.db", block_bytes=1 << 20) as vault:
    ids = vault.ensure("ids", "i64")
    vectors = vault.ensure("embeddings", "vec:f32:384")
    ids.append_array(np.arange(10_000, dtype=np.int64))
    vectors.append_array(np.zeros((10_000, 384), dtype=np.float32))

    batch = vectors.read_array(100, 256)
    selected = vectors.take([5, 100, -1, 5])
    for batch in vectors.iter_batches(batch_bytes=1 << 20):
        print(batch.shape)
```

Numeric input remains contiguous through the native storage path. Input conversion creates stable owned bytes before releasing the GIL, rather than retaining an unsafe reference to the caller's mutable array. Noncontiguous arrays and differing byte orders are converted as needed.

`read_array()` returns a read-only NumPy view backed by owned Python bytes. It remains valid after closing the vault. `copy=True` returns a writable copy. This avoids per-scalar Python objects, but is **not disk-to-NumPy zero-copy**: SQLite fills native memory and the binding copies that output into Python bytes.

Fixed-shape vectors store raw contiguous values, with dtype and shape in metadata and no per-vector type byte. These are storage columns, not similarity-search indexes.

## Scalar convenience and iteration

```python
with ColumnVault("metrics.db") as vault:
    latency = vault.ensure("latency", "f64")
    with latency.cache(cap_bytes=4 << 20, flush_threshold=1 << 20):
        for value in (1.2, 2.4, 3.6):
            latency.append(value)
    print(latency[-1], latency[:2], list(latency))
```

`append`, `extend`, scalar indexing, ordinary slices, and iteration remain available. Iteration has no total-row limit. For sustained numeric throughput, prefer `append_array()`, `read_array()`, and `iter_batches()` instead of scalar calls or Python lists.

Slice steps other than one are rejected. `take(indices)` is the explicit gather API. Fixed-width gathers group requests by block and preserve order, duplicates, and negative indices. Variable-width gathers currently perform scalar reads and are a convenience path.

## Variable-width records

```python
with ColumnVault("records.db", block_bytes=256 << 10) as vault:
    messages = vault.ensure("messages", "str:utf8")
    blobs = vault.ensure("blobs", "bytes")
    records = vault.ensure("records", "msgpack")
    messages.extend(["first", "second", ""])
    blobs.extend([b"image bytes", b"", b"audio bytes"])
    records.extend([{"id": 1}, {"id": 2, "tags": ["train"]}])
    payload, offsets = blobs.read_buffers()
    first = payload[int(offsets[0]):int(offsets[1])]
```

Variable-width blocks contain a payload and little-endian u64 offsets. Empty records are valid. Appending reuses a tail with available capacity; full tails are sealed, not repeatedly resized. A record larger than the configured block size receives a dedicated block. An individual record is limited to 512 MiB.

`read_buffers()` returns `(bytes, numpy.ndarray)` without constructing one Python object per record. `read()` and slicing decode lists. MessagePack, CBOR, and arbitrary-shape tensors use DataPacker and retain per-record serialization costs.

## Types

| Category | Examples |
|---|---|
| Numeric | `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64` |
| Fixed-shape tensors | `vec:f32:384`, `vec:u8:3:224:224` |
| Variable bytes/strings | `bytes`, `str`, `str:utf8`, `str:ascii`, `str:latin1`, `str:utf16le` |
| Fixed-width encoded values | `bytes:128`, `str:32:utf8`, `msgpack:256`, `cbor:512` |
| Variable serialized values | `msgpack`, `cbor`, `vec:f32` |

Fixed-width encoded types retain DataPacker padding rules and reject oversized values. `ensure(name, dtype)` validates the existing dtype. Column numeric codecs are independent of DataPacker, so column improvements do not change KV encodings.

## Visibility and concurrency

Direct append/extend commits payload, offsets, block directory, and length in one transaction. Cached appends become visible to independent readers only on flush. Reading from the buffering column itself first flushes its pending values. `len(column)` includes that handle's pending rows; catalog listings report committed lengths.

```python
with ColumnVault("live.db") as writer:
    column = writer.ensure("step", "i64")
    with ColumnVault("live.db", read_only=True) as reader:
        column.extend([1, 2, 3])
        print(reader["step"].read_array())
```

Each native read uses one short snapshot for metadata and payloads. File-backed vaults have independent reader connections and a serialized writer connection. Blocking native work releases the GIL. Existing reader handles obtain fresh committed lengths on later calls. In-memory vaults serialize access through one connection.

A batch iterator captures an initial end row but reads each batch in a fresh snapshot. It is not an immutable snapshot across concurrent updates/deletions; append-only consumption is the intended case. SQLite still has one active write transaction. Direct appends acquire that reservation before reading append positions.

There is no public multi-column transaction API. Do not perform concurrent structural read-modify-write operations on the same column. `ColumnVault(kv)` shares a file path, not the KV connection or transaction; `:memory:` handles therefore do not share a database.

## Cache and lifecycle

Fixed-width caches use one byte buffer. Variable caches add an offsets buffer; empty records count toward capacity. Oversized batches bypass the cache after existing data is flushed. Failed transactions preserve the pending buffer for retry.

`with vault.cache(...)` covers existing and newly created columns. Nested cache contexts restore prior settings. Context exit flushes even when the body raises; it is not a rollback transaction.

`enable_cache(flush_interval=...)` optionally starts a timer thread inside the application. No external daemon/service is required. Timer errors surface on a subsequent operation. `lock_cache()` pauses timer flushes only, not threshold flushes, and does not make multiple columns atomic.

Use `close()` or a context manager to flush and release handles, especially on Windows. Closing invalidates column handles. A failed flush/close remains retryable. `checkpoint()` flushes caches and returns SQLite's `(busy, log_frames, checkpointed_frames)` result; checkpointing and committing are separate operations.

## Tuning and limitations

The default block target is 1 MiB. Fixed blocks contain an integral number of records, with at least one record. Variable blocks also have a bounded offset capacity. Older `min_chunk_bytes`/`max_chunk_bytes` constructor names are sizing aliases, not the old growth strategy.

For numeric iteration, `batch_bytes` determines the row batch size. For variable iteration it is an estimate using 256 bytes per row, not a strict memory ceiling. Set `batch_size` explicitly for large records or request bounded ranges with `read_buffers()`.

Fixed-width updates overwrite addressed BLOB ranges. Insertions, deletions, and variable-width replacements currently materialize the column and atomically replace its blocks. These are slower convenience operations, not a large-dataset mutation fast path. Compression, null bitmaps, SQL predicate execution, and legacy-column migration are not implemented.

Run the basic benchmark after a release build:

```bash
maturin develop --release
python examples/benchmark_columnar.py --rows 1000000
```

See [storage design](columnar_storage.md) and [performance notes](performance.md) for the physical layout and measurements.
