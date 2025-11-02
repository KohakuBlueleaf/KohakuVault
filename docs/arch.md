# KohakuVault Architecture

## Overview

KohakuVault is a **hybrid Python/Rust storage library** providing two complementary interfaces over a single SQLite database:

1. **KVault** - Key-value store for binary blobs (images, videos, files)
2. **ColumnVault** - Columnar store for typed sequences (timeseries, logs, arrays)

## Why SQLite?

### Single-File Deployment

**Benefit**: Zero configuration, zero external dependencies

```
app.db  ← Everything in one file
├── KVault data (kvault table)
└── ColumnVault data (col_meta + col_chunks tables)
```

**Alternatives require:**
- Redis/memcached: Separate service, network calls, serialization
- LMDB/RocksDB: Separate files, complex C APIs, manual memory management
- Postgres/MySQL: Heavy server, authentication, network overhead

**SQLite gives you:**
- Single file: `app.db`
- Copy = backup: `cp app.db backup.db`
- Deploy anywhere: Just ship the file
- No server, no ports, no auth

### SWMR (Single-Writer Multiple-Reader)

SQLite's **WAL mode** enables concurrent access:

```python
# Process 1: Write
kv = KVault("app.db")
kv["key1"] = b"data"

# Process 2: Read (simultaneously)
kv2 = KVault("app.db")  # No blocking!
val = kv2["key1"]
```

**How WAL works:**
- Writes go to `app.db-wal` (Write-Ahead Log)
- Readers see consistent snapshot
- No read-write blocking (unlike rollback journal)
- Automatic checkpointing merges WAL into main DB

**Perfect for:**
- Background workers + API servers
- Multiple processes accessing same data
- Read-heavy workloads with occasional writes

### Portability

**Cross-platform binary format:**
- Linux ↔ Windows ↔ macOS: Same file works everywhere
- Little-endian/big-endian: SQLite handles it
- 32-bit ↔ 64-bit: Compatible

**Version stability:**
- SQLite format backward compatible
- Database created in 2004 still works in 2025
- Forward compatible with pragmas

## Architecture Layers

```
┌────────────────────────────────────────────────────────────┐
│                   Python User Code                         │
└───────────┬─────────────────────────────┬──────────────────┘
            │                             │
┌───────────▼──────────┐     ┌────────────▼─────────────────┐
│   KVault (proxy.py)  │     │ ColumnVault (column_proxy.py)│
│   - Dict interface   │     │ - Column (list interface)    │
│   - Streaming        │     │ - VarSizeColumn              │
│   - Caching          │     │ - Type packing               │
│   - Retry logic      │     │ - Index calculation          │
└───────────┬──────────┘     └────────────┬─────────────────┘
            │                             │
┌───────────▼──────────┐     ┌────────────▼─────────────────┐
│ _KVault (lib.rs)     │     │ _ColumnVault (col.rs)        │
│ - SQLite ops         │     │ - Chunked storage            │
│ - BLOB streaming     │     │ - Dynamic growth             │
│ - Write-back cache   │     │ - Prefix sum index           │
│ - Thread-safe        │     │ - Range reads/writes         │
└───────────┬──────────┘     └────────────┬─────────────────┘
            │                             │
            └─────────────┬───────────────┘
                          │
                ┌─────────▼────────────┐
                │   rusqlite (Rust)    │
                │   - SQLite bindings  │
                │   - BLOB API         │
                │   - Transactions     │
                └─────────┬────────────┘
                          │
                ┌─────────▼────────────┐
                │   SQLite Database    │
                │   - ACID guarantees  │
                │   - WAL mode (SWMR)  │
                │   - Bundled library  │
                └──────────────────────┘
```

## Design Decisions

### 1. Why Rust Core?

**Rust advantages:**
- **Memory safety**: No segfaults, no data races
- **Performance**: Zero-cost abstractions, native speed
- **SQLite integration**: rusqlite crate is mature and safe
- **PyO3**: Excellent Python bindings, low overhead

**What Rust handles:**
- Low-level SQLite operations
- Binary I/O and BLOB streaming
- Thread-safe connection management
- Memory management

**What Python handles:**
- High-level API (dict/list interfaces)
- Type packing/unpacking (struct module)
- Error mapping and retry logic
- Streaming abstractions

### 2. Why Python Wrapper?

**Python is ideal for:**
- **Ergonomic APIs**: Dict and list syntax feels natural
- **Dynamic typing**: Flexible, easy to use
- **Rich ecosystem**: Works with numpy, pandas, etc.
- **Rapid iteration**: Fast development for proxy layer

**Result**: Best of both worlds
- Rust: Speed, safety, low-level control
- Python: Usability, flexibility, ecosystem

### 3. Why Two Interfaces (KVault + ColumnVault)?

**Different use cases, different optimizations:**

| Aspect | KVault | ColumnVault |
|--------|--------|-------------|
| **Data model** | Unstructured | Structured |
| **Access pattern** | Random key lookup | Sequential + indexed |
| **Typical data** | Images, videos, files | Numbers, logs, vectors |
| **Optimization** | Large BLOB streaming | Chunked array access |
| **Growth** | Fixed per key | Dynamic per column |

**Why not unified?**
- Simpler implementations (single responsibility)
- Clearer APIs (dict vs list semantics)
- Independent optimizations
- Easier to reason about

**Sharing database:**
```python
kv = KVault("app.db")
cv = ColumnVault(kv)  # Same file, different tables

kv["blob"] = bytes    # Unstructured
cv["sequence"][0]     # Structured
```

**Recommended Pattern for Many Large Binaries:**

For applications handling thousands/millions of large binary files (images, videos, documents):

```python
# DON'T: Store all binaries directly in KV
for i in range(1000000):
    vault[f"image:{i}"] = large_binary  # ❌ Inefficient for iteration/filtering

# DO: Use hybrid approach
cv.create_column("image_ids", "i64")
cv.create_column("image_metadata", "bytes")  # JSON, names, etc.

ids = cv["image_ids"]
metadata = cv["image_metadata"]

for img_id, binary, meta in stream:
    ids.append(img_id)           # Columnar: Fast append, efficient scan
    metadata.append(meta)
    kv[f"blob:{img_id}"] = binary  # KV: Optimized for large blobs

# Query metadata without loading binaries
for i in range(len(ids)):
    meta = json.loads(metadata[i])
    if meta["size"] > threshold:
        # Load binary only when needed
        binary = kv[f"blob:{ids[i]}"]
```

**Benefits:**
- Metadata operations (scan, filter) don't load binaries
- Columnar append is O(1) amortized (dynamic chunks)
- KV streaming handles multi-GB files efficiently
- Single SQLite file for both
- Can rebuild indexes/metadata without touching binaries

### 4. Why Chunked Storage for Columns?

**Problem**: Storing millions of elements as one BLOB
- Single huge BLOB is hard to manage
- Can't do partial reads/writes efficiently
- Memory pressure on large columns

**Solution**: Split into dynamically-sized chunks
- Default: 128KB → 16MB per chunk (exponential growth)
- Enables efficient random access
- Partial reads/writes without loading entire column
- Better cache locality

**Dynamic growth (like C++ std::vector):**
```
Chunk lifecycle:
128KB (create) → 256KB (double) → 512KB → 1MB → 2MB → 4MB → 8MB → 16MB (max)
When full at 16MB → Grow to max, create new chunk at 128KB

Critical invariant: All chunks 0..k-1 are ALWAYS at max_chunk_bytes (fully filled)
Only the last chunk k can be < max_chunk_bytes
```

**Benefits:**
- Small footprint initially (128KB)
- Fast amortized O(1) append (same as Python list)
- Bounded max size (good for caching)
- Fewer chunks than fixed-size strategy
- Simple addressing: byte_offset = chunk_idx × max_chunk_bytes + offset_in_chunk

**Cross-chunk element handling:**
- Elements can span chunk boundaries (e.g., 10-byte element at byte 1020 in 1024-byte chunk)
- Byte-based addressing automatically handles this
- Read/write operations seamlessly work across chunks

### 5. Why Prefix Sum for Variable-Size?

**Problem**: Variable-length elements need index

```
[b"short", b"this is longer", b"x", b"medium length"]
 ^         ^                  ^     ^
 Where are the boundaries?
```

**Solution**: Prefix sum index

```
Data column:    [s][h][o][r][t][t][h][i][s][ ][i][s][ ][l][o][n][g][e][r][x][m][e][d][i][u][m]...
Index column:   [5, 19, 20, 33]  ← Cumulative byte offsets
                 │   │   │   │
                 │   │   │   └─ Element 3 ends at byte 33
                 │   │   └───── Element 2 ends at byte 20
                 │   └───────── Element 1 ends at byte 19
                 └───────────── Element 0 ends at byte 5

To get element i:
    start = (i == 0) ? 0 : index[i-1]
    end = index[i]
    data = data_column[start:end]
```

**Complexity:**
- Access element i: O(1) - Two reads (index lookup + data fetch)
- Append: O(1) - Append to data + append to index
- Space overhead: 8 bytes per element (i64 offset)

## Thread Safety

### KVault

**Thread-safe via rusqlite + Mutex:**

```rust
struct _KVault {
    conn: Mutex<Connection>,  // Only one thread at a time
    ...
}
```

**Python-level:**
- Safe for multi-threaded Python (GIL + Rust Mutex)
- Safe for multi-process (SQLite WAL mode)

### ColumnVault

**Same strategy:**

```rust
struct _ColumnVault {
    conn: Mutex<Connection>,
    ...
}
```

**Concurrent access:**
- Multiple processes can read simultaneously (WAL mode)
- Writes are serialized (SQLite-level locking)
- Retry logic handles transient `SQLITE_BUSY` errors

## Data Flow Examples

### KVault Write Path (with Smart Caching)

```
Python: vault["key"] = b"value"
  ↓
proxy.py: __setitem__ → put()
  ↓
lib.rs: _KVault.put()
  ↓
Cache enabled?
  ├─ No → Direct write to SQLite
  └─ Yes ↓
      ↓
      Try cache.insert(key, value)
      ├─ Ok → Cached successfully
      │   ↓
      │   Threshold reached?
      │   └─ Yes → Auto-flush cache
      │
      ├─ ValueTooLarge (value > cap_bytes)
      │   ↓
      │   flush_cache()  // Flush existing
      │   write_direct(key, value)  // Bypass cache
      │
      └─ NeedFlush (would exceed cap_bytes)
          ↓
          flush_cache()  // Make space
          cache.insert(key, value)  // Retry (succeeds)
  ↓
SQL: Batched INSERT (on flush) or direct INSERT
  ↓
SQLite: Write to WAL
  ↓
Disk: app.db-wal
```

**Cache Safety Features:**
- ✅ Never exceeds capacity (pre-checks before insert)
- ✅ Large values handled (auto-bypass)
- ✅ Auto-flush at threshold
- ✅ Auto-flush on disable_cache()
- ✅ Auto-flush on vault.close()
- ✅ Context manager ensures flush
- ✅ Daemon thread for long-running apps

### ColumnVault Append Path

```
Python: col.append(23.5)
  ↓
column_proxy.py: Column.append()
  ↓
  Pack value: struct.pack("<d", 23.5) → 8 bytes
  ↓
col.rs: _ColumnVault.append_raw()
  ↓
  Loop until all data written:
    ↓
    prepare_append_chunk():
      Check last chunk space_left
      ├─ Enough space? → Return chunk
      ├─ Can double? → Double size, return chunk
      ├─ Has ANY space? → Grow to max, return chunk (fill it!)
      └─ Completely full? → Create new chunk at min_chunk_bytes
    ↓
    Calculate bytes_to_write = min(remaining, chunk_capacity - offset)
    ↓
    blob_open(chunk, writable=true)  // Incremental BLOB I/O
    ↓
    blob.write_at(data, offset)  // Direct write without reading
    ↓
    Update current_offset += bytes_written
    ↓
    Loop if more data remaining
  ↓
  Update col_meta.length
  ↓
SQLite: WAL write
  ↓
Disk: app.db-wal
```

**Key optimizations:**
- Incremental BLOB I/O: No need to read 16MB chunks to append 8 bytes
- Chunk size check: Query `actual_size` column before reading blob
- Exponential growth: Efficiently size new chunks based on remaining data
- Fill-first strategy: Always fill current chunk before creating new one

## Performance Characteristics

### KVault

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `vault[key] = val` | O(1) | SQLite B-tree index |
| `vault[key]` get | O(1) | B-tree lookup |
| `vault.put_file()` | O(n/chunk_size) | Streaming, constant memory |
| Iterate all keys | O(n) | Sequential scan |
| Cache flush | O(batch_size) | Single transaction |

### ColumnVault

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `col.append()` | O(1) amortized | Dynamic growth, rare chunk allocation |
| `col[i]` get | O(1) | Direct chunk calculation |
| `col[i]` set | O(1) | Direct chunk write |
| `col.extend(k items)` | O(k) | Batch append, minimal overhead |
| `insert()` / `del` | O(n) | Shifts elements, avoid in loops |
| Iterate | O(n) | Chunked reads (1000 elem batches) |

## File Layout

```
app.db                  ← Main database file
├── kvault table       ← Key-value pairs
│   Columns: id, key (BLOB UNIQUE), value (BLOB), size
│
├── col_meta table     ← Column metadata
│   Columns: col_id, name (UNIQUE), dtype, elem_size, length,
│            chunk_bytes, min_chunk_bytes, max_chunk_bytes
│
└── col_chunks table   ← Chunked column data
    Columns: col_id, chunk_idx, data (BLOB), actual_size
    Primary Key: (col_id, chunk_idx)

app.db-wal             ← Write-Ahead Log (WAL mode)
app.db-shm             ← Shared memory file (WAL mode)
```

## Scaling Characteristics

### KVault Scaling

**Tested up to:**
- 1M+ keys
- Multi-GB values per key (streaming)
- 100GB+ total database size

**Bottlenecks:**
- Disk I/O for large values
- B-tree depth (logarithmic, negligible until ~100M keys)

### ColumnVault Scaling

**Tested up to:**
- 100K+ elements per column (test suite)
- Multiple columns per database

**Theoretical limits:**
- Elements per column: ~9 quintillion (i64 max)
- Practical limit: Disk space and access time

**Growth example (i64 column):**
- 1M elements: ~8MB (1 chunk)
- 10M elements: ~80MB (5 chunks)
- 100M elements: ~800MB (50 chunks)
- 1B elements: ~8GB (500 chunks)

## Why Not...?

### Why not pure Python + SQLite?

**We tried!**

**Issues:**
- **Slower**: Python struct packing overhead
- **Less safe**: Easy to make mistakes with manual SQL
- **More code**: Lots of boilerplate

**Rust gives us:**
- 10-100x faster for intensive operations
- Compile-time safety (no runtime SQL errors)
- Better error handling
- Cleaner separation of concerns

### Why not pure Rust?

**Python wrapper provides:**
- Familiar syntax (`vault[key]`, `col.append()`)
- Easy integration with Python ecosystem
- Flexible type handling
- Rapid prototyping of new features

### Why not other embedded DBs?

**LMDB:**
- ✅ Fast, memory-mapped
- ❌ Complex API, manual memory management
- ❌ Sparse files can waste space
- ❌ Less portable than SQLite

**RocksDB:**
- ✅ Optimized for SSD, write-heavy
- ❌ Complex C++ API
- ❌ Multiple files (not single-file)
- ❌ Requires tuning for good performance

**Lance:**
- ✅ Modern columnar format for ML/AI
- ✅ Rich indexing (vector search, secondary indexes)
- ❌ **File proliferation on appends**: Each append creates new `.lance` transaction files
- ❌ **MVCC overhead**: Maintains multiple versions, creating data/transaction/version files
- ❌ **Compaction required**: Needs periodic `compact()` to merge fragments into single file
- ❌ **Filesystem overhead**: Thousands of small files in append-heavy workloads
- ❌ More complex than needed for simple sequences

**Why KohakuVault for append-heavy:**
- Single SQLite file, no fragmentation
- No compaction needed (WAL auto-checkpoints)
- Simple deployment (one file, done)
- Filesystem-friendly (no inode exhaustion)

**When to use Lance:**
- ML datasets with versioning needs
- Vector similarity search required
- Rich metadata and indexing
- Append-then-read workloads (not continuous appends)

**SQLite:**
- ✅ Single file
- ✅ ACID transactions
- ✅ Mature, stable, well-documented
- ✅ Built into Python (but we bundle for features)
- ✅ Good-enough performance for most use cases
- ✅ Auto-compaction (WAL checkpointing)

## Trade-offs

### Strengths

✅ **Simplicity**: Single file, no configuration
✅ **Portability**: Works everywhere SQLite works
✅ **Safety**: Rust prevents common bugs
✅ **Pythonic**: Familiar dict/list syntax
✅ **Flexible**: Two interfaces, one database
✅ **ACID**: Transactions, durability guarantees

### Limitations

❌ **Not for massive scale**: Use distributed DBs for petabytes
❌ **Single writer**: One write at a time (SWMR, not MWMR)
❌ **No server features**: No users, permissions, remote access
❌ **Limited query language**: Not SQL query interface (by design)
❌ **Not relational**: No joins, foreign keys (at app level)

### Sweet Spot

**KohakuVault excels at:**
- Single-machine applications
- Desktop apps, CLI tools, data pipelines
- Embedded systems, edge devices
- Prototypes and MVPs
- Data that fits on one disk (TB-scale)

**Not ideal for:**
- Multi-server distributed systems
- Concurrent writes from hundreds of processes
- Complex relational queries
- Real-time analytics across sharded data

## Future Directions

**Planned enhancements:**
- [ ] Compression (zstd, lz4) for columns
- [ ] Numpy/Pandas integration
- [ ] Columnar analytics (sum, mean, aggregations)
- [ ] Multi-column transactions
- [ ] Read-only mode for deployment
- [ ] Encryption at rest
- [ ] Replication utilities

## Comparisons

### vs Redis

| Aspect | KohakuVault | Redis |
|--------|-------------|-------|
| Deployment | Single file | Service + network |
| Persistence | Always | Optional |
| Data size | Disk limited | RAM limited |
| Access | In-process | Network calls |
| Concurrency | SWMR (WAL) | MWMR |
| Use case | Persistent storage | Cache + pub/sub |

**When to use KohakuVault:** Persistent local storage, no network needed
**When to use Redis:** Caching, cross-process messaging, distributed

### vs Shelve (Python stdlib)

| Aspect | KohakuVault | Shelve |
|--------|-------------|--------|
| Backend | SQLite | dbm (platform-dependent) |
| Performance | Fast (Rust) | Slow (Python pickle) |
| Type safety | Bytes only | Any Python object (pickle) |
| Streaming | Yes | No |
| Columnar | Yes | No |
| Portability | Excellent | Poor (dbm varies) |

**When to use KohakuVault:** Production use, performance matters, large files
**When to use Shelve:** Quick prototypes, small data, simplicity

### vs HDF5/Zarr

| Aspect | KohakuVault | HDF5/Zarr |
|--------|-------------|-----------|
| Focus | General storage | Array/numerical data |
| Interface | Dict + List | Array (numpy-like) |
| Metadata | Manual | Rich (attributes, groups) |
| Chunking | Dynamic | Fixed |
| Compression | Planned | Built-in |
| Ecosystem | Growing | Mature (scientific) |

**When to use KohakuVault:** General-purpose storage, mixed data types
**When to use HDF5/Zarr:** Scientific datasets, multi-dimensional arrays

### vs Lance

| Aspect | KohakuVault | Lance |
|--------|-------------|-------|
| File structure | Single SQLite file | Multiple .lance files + manifests |
| Append pattern | Direct WAL writes | Creates new transaction files per append |
| Compaction | Auto (WAL checkpoint) | Manual `compact()` required |
| MVCC | SQLite-level | Full versioning (time-travel) |
| File count | 1-3 files (db + wal + shm) | Hundreds to thousands (.lance + _versions + _transactions) |
| Filesystem | Friendly (few inodes) | Heavy (many small files) |
| Vector search | No | Built-in (ANN indexes) |
| Metadata | Manual | Rich (tags, schemas) |
| Use case | Append-heavy logs/timeseries | ML datasets with versioning |

**Append-heavy workload example (1M appends):**

```
KohakuVault:
├── data.db         (main file)
├── data.db-wal     (write-ahead log, auto-compacted)
└── data.db-shm     (shared memory)
Total: 3 files

Lance:
├── data.lance/_versions/1.manifest
├── data.lance/_versions/2.manifest
├── data.lance/_versions/3.manifest
├── ... (thousands of manifest files)
├── data.lance/_transactions/0-0.txn
├── data.lance/_transactions/0-1.txn
├── ... (thousands of transaction files)
├── data.lance/data/0.lance
├── data.lance/data/1.lance
├── ... (many data fragments)
└── [Needs compact() to merge → single file]
Total: 1000s of files before compaction
```

**When to use KohakuVault:**
- Continuous append streams (logs, sensors, events)
- Simple deployment (single file)
- No need for versioning/time-travel
- Filesystem-constrained environments

**When to use Lance:**
- ML training datasets (versioning helpful)
- Need vector similarity search
- Rich metadata and schemas
- Batch write patterns (not continuous appends)

## Summary

KohakuVault provides **simple, fast, portable storage** by combining:

- **SQLite** for proven persistence and portability
- **Rust** for performance and safety
- **Python** for ergonomic interfaces
- **Dual paradigms** for different data types

**Philosophy**: Maximize simplicity and developer ergonomics while maintaining excellent performance through Rust optimization.
