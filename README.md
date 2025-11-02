# KohakuVault

High-performance, SQLite-backed storage with dual interfaces: **dict-like for blobs** (key-value) and **list-like for sequences** (columnar). Rust core with Pythonic APIs.

## Quick Start
```bash
pip install kohakuvault
```

```python
from kohakuvault import KVault, ColumnVault

# Key-Value: Store binary blobs (images, files, etc.)
kv = KVault("data.db")
kv["image:123"] = image_bytes
kv["video:456"] = video_bytes

# Columnar: Store typed sequences (timeseries, logs, events)
cv = ColumnVault(kv)  # Shares same database
cv.create_column("temperatures", "f64")
cv.create_column("log_messages", "bytes")  # Variable-size strings

temps = cv["temperatures"]
temps.extend([23.5, 24.1, 25.0])  # Like a list

logs = cv["log_messages"]
logs.append(b"Server started")
logs.append(b"Request processed in 5.2ms")

# Access
print(temps[0])      # 23.5
print(list(logs))    # [b'Server started', b'Request processed in 5.2ms']
```

## Features

- **Dual interfaces**: Dict for blobs (KVault), List for sequences (ColumnVault)
- **Zero external dependencies**: Single SQLite file, no services required
- **Memory efficient**: Stream multi-GB files, dynamic chunk growth
- **Type-safe columnar**: Fixed-size (i64, f64, bytes:N) and variable-size (bytes)
- **Rust performance**: Native speed with Pythonic ergonomics

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

```python
vault = KVault("media.db")
vault.enable_cache(cap_bytes=64*1024*1024, flush_threshold=16*1024*1024)

# Writes batched in memory
for i in range(1000):
    vault[f"item:{i}"] = data

vault.flush_cache()  # Commit all at once
vault.disable_cache()
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
pytest                               # Test
git commit && git push
# Open PR
```

## Releasing

GitHub Actions automatically builds wheels and publishes to PyPI when you push a tag:

```bash
# 1. Update version in pyproject.toml and Cargo.toml
# 2. Commit changes
git add pyproject.toml Cargo.toml
git commit -m "Bump version to 0.1.0"

# 3. Create and push tag
git tag v0.1.0
git push origin main --tags

# 4. GitHub Actions will:
#    - Build wheels for all platforms
#    - Create GitHub Release with wheels attached
#    - Publish to PyPI (with skip-existing for safety)
```

**What happens:**
- Wheels are built for Linux, Windows, macOS (Apple Silicon)
- All wheels are uploaded to the GitHub Release (downloadable)
- Wheels are published to PyPI
- If some wheels already exist on PyPI, they're skipped (no error)

## License

Apache 2.0 - see [LICENSE](LICENSE)
