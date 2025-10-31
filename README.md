# KohakuVault

A high-performance, SQLite-backed key-value store for large media files. Rust core with a Pythonic dict-like interface.

## Quick Start

```python
from kohakuvault import KVault

# Dict-like interface for binary data
vault = KVault("media.db")
vault["thumbnail:123"] = image_bytes
vault["video:456"] = video_bytes

# Stream large files without loading into memory
with open("large_video.mp4", "rb") as f:
    vault.put_file("video:789", f)

# Write-back cache for bulk operations
vault.enable_cache(cap_bytes=64*1024*1024)
for i in range(1000):
    vault[f"frame:{i}"] = frame_data
vault.flush_cache()  # Batch commit to disk
```

## Why KohakuVault?

- **No external services**: Single-file SQLite database, no Redis/memcached needed
- **Memory efficient**: Stream multi-GB files without loading into RAM
- **Fast**: Rust implementation with optional write-back caching
- **Pythonic**: Feels like a dict, works like a database

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

GitHub Actions automatically builds and publishes wheels to PyPI on git tags:

```bash
# 1. Update version in pyproject.toml and Cargo.toml
# 2. Create and push tag
git tag v0.1.0 && git push origin v0.1.0
# 3. Wheels auto-build for all platforms and publish to PyPI
```

See [.github/RELEASE.md](.github/RELEASE.md) for detailed release instructions and required secrets setup.

## License

MIT - see [LICENSE](LICENSE)
