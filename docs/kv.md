# KVault API Reference

Complete API documentation for KohakuVault's key-value storage.

## Overview

**KVault** provides a dict-like interface for storing binary blobs (images, videos, documents, etc.) with streaming support and write-back caching.

## Constructor

```python
KVault(
    path,                    # SQLite database path
    chunk_size=1048576,      # Streaming chunk size (1 MiB)
    retries=4,               # Retry attempts for busy DB
    backoff_base=0.02,       # Initial backoff delay (20ms)
    table="kvault",          # Table name
    enable_wal=True,         # Write-Ahead Logging
    page_size=32768,         # SQLite page size (32KB)
    mmap_size=268435456,     # Memory-mapped I/O (256MB)
    cache_kb=100000          # SQLite cache size (100MB)
)
```

### Parameters

- **path** (str | Path): Database file path (created if doesn't exist)
- **chunk_size** (int): Streaming read/write chunk size in bytes
- **retries** (int): Max retry attempts for `SQLITE_BUSY` errors
- **backoff_base** (float): Initial exponential backoff delay in seconds
- **table** (str): SQLite table name (allows multiple KVaults in one DB)
- **enable_wal** (bool): Enable Write-Ahead Logging for SWMR
- **page_size** (int): SQLite page size (set on DB creation only)
- **mmap_size** (int): Memory-mapped I/O size (0 = disabled)
- **cache_kb** (int): SQLite page cache size in KB

## Core Methods

### put(key, value)

Store a value for a key.

```python
vault.put("user:123", b"profile data")
vault.put(b"\x00\x01", image_bytes)  # Binary keys OK
```

**Parameters:**
- **key** (str | bytes): Unique identifier
- **value** (bytes-like): Data to store

**Returns:** None

**Raises:**
- `InvalidArgument`: Invalid key or value type
- `DatabaseBusy`: SQLite locked (after retries)

### put_file(key, reader, size=None, chunk_size=None)

Stream from file-like object to vault.

```python
with open("large_video.mp4", "rb") as f:
    vault.put_file("video:123", f)

# Custom chunk size
with open("big_file.bin", "rb") as f:
    vault.put_file("file:456", f, chunk_size=4*1024*1024)  # 4 MiB chunks
```

**Parameters:**
- **key** (str | bytes): Unique identifier
- **reader** (file-like): Object with `read(n)` method
- **size** (int, optional): Total bytes (auto-detected if not provided)
- **chunk_size** (int, optional): Override default chunk size

**Returns:** None

**Memory usage:** O(chunk_size) - doesn't load entire file into RAM

### get(key, default=None)

Retrieve value for key.

```python
data = vault.get("user:123")            # Returns bytes or None
data = vault.get("missing", b"fallback") # Returns fallback
```

**Parameters:**
- **key** (str | bytes): Identifier
- **default** (Any): Returned if key not found

**Returns:** bytes | default

**Raises:** No exceptions (returns default on error)

### get_to_file(key, writer, chunk_size=None)

Stream from vault to file-like object.

```python
with open("output.mp4", "wb") as f:
    bytes_written = vault.get_to_file("video:123", f)
```

**Parameters:**
- **key** (str | bytes): Identifier
- **writer** (file-like): Object with `write(bytes)` method
- **chunk_size** (int, optional): Override default chunk size

**Returns:** int - Total bytes written

**Raises:**
- `NotFound`: Key doesn't exist

**Memory usage:** O(chunk_size)

### delete(key)

Remove a key.

```python
vault.delete("user:123")  # Returns True if deleted
vault.delete("missing")   # Returns False if not found
```

**Parameters:**
- **key** (str | bytes): Identifier

**Returns:** bool - True if deleted, False if not found

### exists(key)

Check if key exists.

```python
if vault.exists("user:123"):
    print("User exists")
```

**Parameters:**
- **key** (str | bytes): Identifier

**Returns:** bool

## Dict-Like Interface

KVault implements Python's `Mapping` protocol.

### Indexing

```python
# Get (raises KeyError if missing)
data = vault["user:123"]

# Set
vault["user:123"] = b"new data"

# Delete (raises KeyError if missing)
del vault["user:123"]

# Check existence
if "user:123" in vault:
    ...
```

### Mapping Methods

```python
# Length
count = len(vault)

# Keys
for key in vault.keys():
    print(key)

# Values
for value in vault.values():
    print(len(value))

# Items
for key, value in vault.items():
    print(f"{key}: {len(value)} bytes")

# Iterate (over keys)
for key in vault:
    print(key)

# Safe get
data = vault.get("key", default=b"")

# Pop
data = vault.pop("key")  # Returns and deletes

# Set default
vault.setdefault("key", b"default")

# Update
vault.update({"k1": b"v1", "k2": b"v2"})

# Clear all
vault.clear()
```

## Caching Methods

### enable_cache(cap_bytes=67108864, flush_threshold=16777216)

Enable write-back caching for batch writes.

```python
vault.enable_cache(
    cap_bytes=64*1024*1024,        # 64 MiB capacity
    flush_threshold=16*1024*1024   # Auto-flush at 16 MiB
)

# Writes cached in memory
for i in range(10000):
    vault[f"key:{i}"] = b"value"

vault.flush_cache()  # Commit to disk
```

**Parameters:**
- **cap_bytes** (int): Maximum cache size in bytes
- **flush_threshold** (int): Auto-flush when exceeded

**How it works:**
- Writes stored in Rust HashMap (in-memory)
- Auto-flushes when threshold reached
- Manual flush with `flush_cache()`
- Batched in single transaction (fast)

**Use cases:**
- Bulk imports
- Batch processing
- Reducing disk writes

### disable_cache()

Disable cache and flush pending writes.

```python
vault.disable_cache()
```

### flush_cache()

Manually flush cache to disk.

```python
flushed_count = vault.flush_cache()
print(f"Flushed {flushed_count} entries")
```

**Returns:** int - Number of entries flushed

## Maintenance Methods

### optimize()

Run VACUUM and optimize database.

```python
vault.optimize()
```

**What it does:**
- VACUUM: Reclaim unused space, defragment
- PRAGMA optimize: Update query planner statistics

**When to use:**
- After deleting many keys
- Periodically for long-running applications
- Before backup/deployment

**Warning:** Blocks writes during VACUUM (can take minutes for large DBs)

### close()

Flush cache and close connection.

```python
vault.close()
```

**Always close vaults when done**, especially with caching enabled!

### Context Manager

```python
with KVault("data.db") as vault:
    vault.enable_cache()
    vault["key"] = b"value"
    # Auto-flushes and closes on exit
```

## Exceptions

All exceptions inherit from `KohakuVaultError`.

### NotFound

Key doesn't exist.

```python
from kohakuvault import NotFound

try:
    data = vault["missing_key"]
except NotFound as e:
    print(f"Key not found: {e.key}")
```

**Note:** Dict interface raises `KeyError`, not `NotFound`:

```python
# This raises KeyError (Python convention)
data = vault["missing"]

# This raises NotFound
data = vault.get("missing")  # No, returns None
# NotFound only raised internally, converted to KeyError at interface level
```

### DatabaseBusy

SQLite database locked (after retries exhausted).

```python
from kohakuvault import DatabaseBusy

try:
    vault["key"] = b"data"
except DatabaseBusy:
    print("DB locked, retry later")
```

**Usually auto-retried**, but can happen under heavy concurrent load.

### InvalidArgument

Invalid input (wrong types, invalid values).

```python
vault[123] = b"data"  # Raises InvalidArgument (key must be str/bytes)
```

### IoError

File I/O error during streaming.

```python
try:
    vault.put_file("key", broken_file_handle)
except IoError as e:
    print(f"I/O error: {e}")
```

## Underlying Implementation

### Schema

```sql
CREATE TABLE kvault (
    id    INTEGER PRIMARY KEY,
    key   BLOB UNIQUE NOT NULL,
    value BLOB NOT NULL,
    size  INTEGER NOT NULL
);

CREATE UNIQUE INDEX kvault_key_idx ON kvault(key);
```

### Storage

**Small values (< few KB):**
- Stored directly in table row
- Fast access (single B-tree lookup)

**Large values (> few KB):**
- Stored as BLOB
- Streamed via `put_file()` / `get_to_file()`
- Uses SQLite's zeroblob + incremental BLOB I/O

### Caching

**Write-back cache (Rust HashMap):**

```rust
struct WriteBackCache {
    map: HashMap<Vec<u8>, Vec<u8>>,  // key → value
    current_bytes: usize,             // Total cached bytes
    cap_bytes: usize,                 // Max capacity
    flush_threshold: usize,           // Auto-flush threshold
}
```

**On flush:**
- Batched INSERT in single transaction
- Much faster than individual writes
- Atomic (all-or-nothing)

### Retry Logic

**Exponential backoff for transient errors:**

```python
def _with_retries(call, attempts=4, backoff_base=0.02):
    for attempt in range(attempts):
        try:
            return call()
        except DatabaseBusy:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff_base * (2 ** attempt))
```

**Backoff sequence:**
- Attempt 1: 20ms
- Attempt 2: 40ms
- Attempt 3: 80ms
- Attempt 4: Raise exception

## Performance Tips

### Bulk Writes

```python
# Slow: 10,000 individual writes
for i in range(10000):
    vault[f"key:{i}"] = b"data"

# Fast: Cached + batched
vault.enable_cache()
for i in range(10000):
    vault[f"key:{i}"] = b"data"
vault.flush_cache()  # One transaction
```

**Speedup:** ~10-100x depending on data size

### Streaming

```python
# Slow: Load entire file into memory
with open("big.mp4", "rb") as f:
    vault["video"] = f.read()  # Reads all into RAM

# Fast: Stream
with open("big.mp4", "rb") as f:
    vault.put_file("video", f)  # Constant memory
```

**Memory usage:** O(chunk_size) vs O(file_size)

### Tuning

```python
# For SSD with large files
vault = KVault(
    "data.db",
    page_size=8192,           # Smaller pages for random access
    mmap_size=1024*1024*1024, # 1GB memory-mapped
    cache_kb=200000           # 200MB cache
)

# For HDD with sequential access
vault = KVault(
    "data.db",
    page_size=65536,          # Larger pages (64KB)
    chunk_size=4*1024*1024,   # Larger chunks (4MB)
    cache_kb=50000            # 50MB cache
)
```

## Examples

See `examples/basic_usage.py` for:
- Basic put/get operations
- Streaming large files
- Write-back caching
- Context managers
- Custom configuration

## See Also

- `docs/arch.md` - Overall architecture
- `docs/col.md` - Columnar storage API
- `examples/basic_usage.py` - Working examples
