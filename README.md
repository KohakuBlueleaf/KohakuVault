# KohakuVault

High-performance, SQLite-backed storage with dual interfaces: **dict-like for blobs** (key-value) and **list-like for sequences** (columnar). Rust core with Pythonic APIs.

## Quick Start

```bash
pip install kohakuvault
```

**KV Store** - Dict-like interface for binary blobs (images, videos, documents):

```python
from kohakuvault import KVault

vault = KVault("data.db")
vault["image:123"] = image_bytes
vault["video:456"] = video_bytes
data = vault["image:123"]

# Bulk writes with smart caching (NEW in v0.2.2!)
with vault.cache(64*1024*1024):
    for i in range(10000):
        vault[f"key:{i}"] = data
# Auto-flushes on exit!
```

**Columnar Storage** - List-like interface for typed sequences (timeseries, logs, events):

```python
from kohakuvault import ColumnVault

cv = ColumnVault("data.db")

# Primitives
cv.create_column("temperatures", "f64")
temps = cv["temperatures"]
temps.extend([23.5, 24.1, 25.0])
print(temps[0])  # 23.5

# High-performance bulk writes with cache (NEW in v0.4.1!)
with temps.cache():
    for temp in sensor_readings:
        temps.append(temp)  # 10-100x faster!

# Structured data (NEW in v0.3.0!)
cv.create_column("users", "msgpack")
users = cv["users"]
users.append({"name": "Alice", "age": 30, "tags": ["vip"]})
print(users[0])  # {'name': 'Alice', 'age': 30, 'tags': ['vip']}

# Strings with encoding (NEW in v0.3.0!)
cv.create_column("messages", "str:utf8")
messages = cv["messages"]
messages.append("Hello, 世界!")
print(messages[0])  # 'Hello, 世界!'
```

**DataPacker** - Rust-based serialization (NEW in v0.3.0!):

```python
from kohakuvault import DataPacker

# MessagePack for structured data
packer = DataPacker("msgpack")
packed = packer.pack({"user": "alice", "score": 95.5})
data = packer.unpack(packed, 0)

# Bulk operations
records = [{"id": i, "val": i*1.5} for i in range(1000)]
packed_all = packer.pack_many(records)  # Concatenated bytes

# Unpack with offsets (for variable-size)
offsets = [0, len(packer.pack(records[0]))]  # Calculate offsets
unpacked = packer.unpack_many(packed_all, offsets=offsets)
```

## Features

- **Dual interfaces**: Dict for blobs (KVault), List for sequences (ColumnVault)
- **Zero external dependencies**: Single SQLite file, no services required
- **Memory efficient**: Stream multi-GB files, dynamic chunk growth
- **Type-safe columnar**: Fixed-size (i64, f64, bytes:N) and variable-size (bytes, str, msgpack, cbor)
- **Rust performance**: Native speed with Pythonic ergonomics
- **Smart caching**: Write-back cache for 10-100x faster bulk writes (NEW in v0.4.1!)
- **Structured data**: Store dicts/lists directly with MessagePack/CBOR (v0.3.0)
- **DataPacker**: Rust-based serialization with multi-encoding support (v0.3.0)

## Best Practices

### Handling Many Large Binary Files

**For thousands of large binaries (images, videos, documents), use a hybrid approach:**

```python
from kohakuvault import KVault, ColumnVault

kv = KVault("media.db")
cv = ColumnVault(kv)  # Share same database

# Store metadata in columnar (efficient for large lists)
cv.create_column("image_ids", "i64")
cv.create_column("image_names", "bytes")
cv.create_column("image_sizes", "i64")
cv.create_column("upload_times", "i64")

ids = cv["image_ids"]
names = cv["image_names"]
sizes = cv["image_sizes"]
times = cv["upload_times"]

# Store actual binaries in KV store
for img_id, img_data, img_name in image_stream:
    # Metadata in columnar (fast append, efficient iteration/filtering)
    ids.append(img_id)
    names.append(img_name)
    sizes.append(len(img_data))
    times.append(int(time.time()))

    # Binary data in KV (optimized for large blobs)
    kv[f"blob:{img_id}"] = img_data

# Query metadata without loading binaries
for i in range(len(ids)):
    if sizes[i] > 1024 * 1024:  # Find images > 1MB
        print(f"Large image: {names[i].decode()}")
        # Load binary only when needed
        data = kv[f"blob:{ids[i]}"]
```

**Why this pattern?**
- ✅ Columnar optimized for append-heavy metadata (millions of entries)
- ✅ KV optimized for large binary blobs (streaming, caching)
- ✅ Can query/filter metadata without loading binaries
- ✅ Both share same SQLite file (single-file deployment)
- ✅ Efficient iteration over metadata, lazy loading of binaries

## Installation

```bash
pip install kohakuvault  # When published to PyPI
pip install .            # From source
```

**Platform Support**:
- ✅ Linux (x86_64)
- ✅ Windows (x86_64)
- ✅ macOS (Apple Silicon M1/M2/M3/M4 only - ARM64)
- ❌ macOS Intel (x86_64) - not supported

## Development

**Prerequisites**: Python 3.10+, Rust ([rustup.rs](https://rustup.rs/))

```bash
# Setup
git clone https://github.com/yourusername/kohakuvault.git
cd kohakuvault
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev]
maturin develop  # Build Rust extension (once)

# Workflow
# - Edit Python files → changes live immediately
# - Edit Rust files → run `maturin develop` to rebuild

# Tools
pytest                  # Run tests
black src/kohakuvault   # Format Python
cargo fmt               # Format Rust
maturin build --release # Build production wheel
```

## Usage

### Basic Operations

```python
vault = KVault("media.db")

# Dict-like interface
vault["key"] = b"value"
data = vault["key"]
del vault["key"]
if "key" in vault: ...

# Safe retrieval
data = vault.get("key", default=b"")

# Iteration
for key in vault:
    print(f"{key}: {len(vault[key])} bytes")
```

### Streaming Large Files

```python
vault = KVault("media.db", chunk_size=1024*1024)  # 1 MiB chunks

# Stream from file → vault
with open("large_video.mp4", "rb") as f:
    vault.put_file("video:789", f)

# Stream from vault → file
with open("output.mp4", "wb") as f:
    vault.get_to_file("video:789", f)
```

### Bulk Operations with Caching

**Recommended: Use context manager for automatic flush**

```python
vault = KVault("media.db")

# Safest: Context manager auto-flushes
with vault.cache(cap_bytes=64*1024*1024):
    for i in range(1000):
        vault[f"item:{i}"] = data
# Auto-flushed here, guaranteed!

# Long-running: Daemon thread auto-flushes every 5 seconds
vault.enable_cache(cap_bytes=64*1024*1024, flush_interval=5.0)
while True:
    vault["sensor_data"] = read_sensor()
# Daemon flushes automatically

# Manual control (backward compatible)
vault.enable_cache(cap_bytes=64*1024*1024)
for i in range(1000):
    vault[f"item:{i}"] = data
vault.flush_cache()  # Manual flush
vault.disable_cache()  # Auto-flushes before disabling
```

### Configuration

```python
vault = KVault(
    path="media.db",
    chunk_size=2*1024*1024,   # Streaming chunk size
    retries=10,                # Retry attempts for busy DB
    enable_wal=True,           # Write-Ahead Logging
    cache_kb=20000,            # SQLite cache size
)
```

### Columnar Storage (NEW!)

List-like interface for typed sequences (timeseries, logs, events):

```python
from kohakuvault import ColumnVault

cv = ColumnVault("data.db")

# Fixed-size types: i64, f64, bytes:N
cv.create_column("sensor_temps", "f64")
cv.create_column("timestamps", "i64")
cv.create_column("hashes", "bytes:32")  # 32-byte fixed

temps = cv["sensor_temps"]
temps.append(23.5)
temps.extend([24.1, 25.0, 25.3])
print(temps[0], temps[-1], len(temps))  # 23.5, 25.3, 4

# Variable-size bytes (for strings, JSON, etc.)
cv.create_column("log_messages", "bytes")  # No size = variable!
logs = cv["log_messages"]
logs.append(b"Short message")
logs.append(b"This is a much longer log entry with details...")
print(logs[0])  # Exact bytes, no padding

# Iterate
for temp in temps:
    print(temp)
```

**Why columnar?**
- Append-heavy workloads (O(1) amortized, like Python list)
- Typed data (int/float/bytes with type safety)
- Efficient iteration and random access
- Dynamic chunk growth (128KB → 16MB, exponential like std::vector)
- Cross-chunk element support (byte-based addressing)
- Minimal memory overhead (incremental BLOB I/O)

See `docs/COLUMNAR_GUIDE.md` and `examples/columnar_demo.py` for complete guide.

## API Reference

### Constructor

```python
KVault(path, chunk_size=1048576, retries=4, backoff_base=0.02,
       table="kvault", enable_wal=True, page_size=4096,
       mmap_size=268435456, cache_kb=20000)
```

### Methods

**Storage**
- `put(key, value)` - Store bytes
- `put_file(key, reader, size=None, chunk_size=None)` - Stream from file-like
- `get(key, default=None)` - Retrieve bytes
- `get_to_file(key, writer, chunk_size=None)` - Stream to file-like
- `delete(key)` - Remove key
- `exists(key)` - Check existence

**Caching**
- `enable_cache(cap_bytes, flush_threshold)` - Enable write-back cache
- `disable_cache()` - Disable and flush cache
- `flush_cache()` - Commit cached writes, returns count

**Maintenance**
- `optimize()` - VACUUM database
- `close()` - Flush and close

**Dict Interface**: `vault[key]`, `del vault[key]`, `key in vault`, `len(vault)`, `vault.keys()`, `vault.values()`, `vault.items()`, etc.

**Exceptions**: `KohakuVaultError`, `NotFound`, `DatabaseBusy`, `InvalidArgument`, `IoError`

## Architecture

```
Python wrapper (src/kohakuvault/proxy.py)
    ↓ PyO3 bindings
Rust core (src/kvault-rust/lib.rs)
    ↓ rusqlite
SQLite database (bundled)
```

**Why hybrid?** Rust handles SQLite operations safely and efficiently. Python provides the ergonomic dict-like interface.

## Contributing

```bash
# Setup
git checkout -b feature-name
# Make changes
black src/kohakuvault && cargo fmt  # Format
cargo clippy                        # Basic linting
pytest                              # Test
git commit && git push
# Open PR
```

## License

Apache 2.0 - see [LICENSE](LICENSE)
