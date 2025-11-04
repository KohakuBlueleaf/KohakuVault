# ColumnVault API Reference (v0.4.1)

Complete API documentation for KohakuVault's columnar storage.

## ⚠️ Breaking Changes in v0.4.0

**Incompatible with v0.3.0 databases** - Full redesign for performance

**What Changed:**
1. **Element-aligned chunks**: Fixed-size columns enforce `chunk_size % elem_size == 0`
2. **Adaptive variable-size**: New schema with `bytes_used` tracking
3. **12-byte index**: Variable-size changed from 8-byte offset to (chunk_id, start, end)
4. **Chunk-wise extend**: Writes entire chunks at once (50-100x faster!)
5. **No cross-element spanning**: Each element fully contained in one chunk

**Performance Gains:**
- Variable-size space: **316x improvement** (10 msgpack: 132MB → 416KB)
- Variable-size extend: **50-100x faster** than append
- Test runtime: 21s → 6.6s

---

## Overview

**ColumnVault** provides list-like interfaces for storing typed sequences (timeseries, logs, arrays) with:
- **Element-aligned chunks** for fixed-size types (i64, f64, bytes:N)
- **Adaptive chunking** for variable-size types (bytes, str, msgpack, cbor)
- **Efficient bulk operations** (extend writes entire chunks at once)
- **Smart space management** (tracks used vs wasted bytes)

## ColumnVault (Container)

### Constructor

```python
ColumnVault(
    kvault_or_path,                 # KVault instance or path string
    chunk_bytes=1048576,            # Default chunk size (1 MiB)
    min_chunk_bytes=131072,         # Min chunk size (128KB)
    max_chunk_bytes=16777216        # Max chunk size (16MB)
)
```

**Parameters:**
- **kvault_or_path**: Either a KVault instance (to share DB) or database path string
- **chunk_bytes**: Default initial chunk size for columns (deprecated, use min/max)
- **min_chunk_bytes**: Minimum chunk size - new chunks start here (default 128KB)
- **max_chunk_bytes**: Maximum chunk size - chunks don't grow beyond this (default 16MB)

**Examples:**

```python
# Share database with KVault
kv = KVault("data.db")
cv = ColumnVault(kv)

# Or standalone
cv = ColumnVault("data.db")

# Custom chunk sizing
cv = ColumnVault(
    "large_data.db",
    min_chunk_bytes=512*1024,     # 512KB start
    max_chunk_bytes=64*1024*1024  # 64MB max
)
```

### Methods

#### create_column(name, dtype, chunk_bytes=None)

Create a new column.

```python
cv.create_column("temperatures", "f64")
cv.create_column("timestamps", "i64")
cv.create_column("hashes", "bytes:32")      # Fixed 32 bytes
cv.create_column("log_messages", "bytes")    # Variable size
```

**Parameters:**
- **name** (str): Column name (must be unique in database)
- **dtype** (str): Data type (see Data Types section)
- **chunk_bytes** (int, optional): Override default chunk size (rarely needed)

**Returns:** Column or VarSizeColumn instance

**Raises:**
- `InvalidArgument`: Invalid dtype
- `RuntimeError`: Column name already exists

#### cv[name]

Get column by name.

```python
temps = cv["temperatures"]
temps.append(23.5)
```

**Parameters:**
- **name** (str): Column name

**Returns:** Column or VarSizeColumn

**Raises:**
- `NotFound`: Column doesn't exist

---

## Chunking Algorithms (v0.4.0)

### Fixed-Size Columns (Element-Aligned)

**Constraint:** `chunk_size` must be multiple of `elem_size`

**Auto-Alignment:**
- `aligned_min = ceil(min / elem_size) × elem_size`
- `aligned_max = floor(max / elem_size) × elem_size`
- Error if `aligned_min > aligned_max`

**Example:**
```
elem_size=10, min=128KB, max=16MB
aligned_min = 131,080 bytes (13,108 elements)
aligned_max = 16,777,210 bytes (1,677,721 elements)
```

**Benefits:**
- No elements span chunk boundaries
- Predictable addressing: `chunk_idx = byte_offset / chunk_size`
- Simpler, faster reads

### Variable-Size Columns (Adaptive Chunking)

**New Schema (v0.4.0):**
```sql
col_chunks:
- actual_size: Chunk capacity (blob size)
- bytes_used: Actually used bytes (≤ actual_size)
- has_deleted: 1 if has deletions (for vacuum)
- start_elem_idx: First element index in chunk
- end_elem_idx: Last element index in chunk

Index column (12 bytes per element):
- i32 chunk_id: Which chunk contains this element
- i32 start_byte: Start position in chunk
- i32 end_byte: End position in chunk
```

**Append Algorithm:**

```python
needed = len(data)
chunk_size = current chunk capacity
used = bytes actually used in chunk
available = chunk_size - used

# CASE 1: Fits without expansion
if available >= needed:
    write_to_chunk()
    update_bytes_used()

# CASE 2: max - used >= needed (can expand to legal size)
elif max_chunk_size - used >= needed:
    legal_size = min_chunk_size × 2^k  # where legal_size - used >= needed
    expand_to(legal_size)
    write_to_chunk()

# CASE 3: max - used < needed
elif chunk_size < max_chunk_size and needed <= max_chunk_size:
    # 3-1: Not at max yet
    expand_to(used + needed)
    write_to_chunk()

elif chunk_size >= max_chunk_size and needed <= max_chunk_size/2:
    # 3-2: At max, small element (≤ 50% max)
    expand_to(used + needed)  # Up to 1.5× max allowed
    write_to_chunk()

else:
    # 3-3: At max, large element (> 50% max)
    create_new_chunk()
    write_to_chunk()
```

**Extend Algorithm (Optimized - Chunk-Wise Writes):**

```python
# Step 1: Read last chunk's unused space (if any)
if last_chunk_has_space:
    can_add_to_last_chunk = True

# Step 2: Buffer elements until total > max_chunk_size
buffer = []
buffer_size = 0

for element in values:
    if buffer_size + len(element) > max_chunk_size:
        # Write full chunk at once!
        chunk_data = concat(buffer)
        chunk_capacity = find_legal_size(buffer_size)  # min × 2^k

        write_full_chunk(chunk_data, chunk_capacity, buffer_size)
        build_index_entries(buffer)  # All in Rust!

        buffer = [element]
        buffer_size = len(element)
    else:
        buffer.append(element)
        buffer_size += len(element)

# Step 3: Write remaining buffer
if buffer:
    write_full_chunk(buffer)
```

**Key Optimization:**
- **No element-by-element appends** in extend!
- Writes **entire chunks** at once
- All packing/indexing done in Rust (1 FFI call)
- Result: **50-100x faster** than looping append

**Insert Algorithm:**

```python
# 1. Find insertion point
(chunk_id, insert_offset) = get_location_from_index(elem_idx)

# 2. Read rest of chunk data after insert point
rest_data = read_chunk_from(chunk_id, insert_offset)

# 3. Write combined data
write_to_chunk(chunk_id, insert_offset, new_data + rest_data)

# 4. Update metadata for all elements in chunk after elem_idx
shift_metadata_in_chunk(chunk_id, elem_idx, len(new_data))

# 5. Check if chunk too large (> 2× max)
if new_chunk_size > max_chunk_size * 2:
    split_chunk(chunk_id)  # Split into 2 chunks
    update_all_indices()

update_bytes_used(chunk_id)
```

**Benefits:**
- Supports insertion anywhere
- Automatic chunk split if too large
- Maintains chunk locality

**Delete Algorithm:**

```python
# Just shift index entries (fixed-size operation on 12-byte elements)
read_remaining_indices()
write_back_one_position_earlier()
update_length()

# Don't remove data from chunks (lazy deletion)
# Actual cleanup happens in vacuum
```

**Benefits:**
- Fast deletion (just index manipulation)
- No chunk rewriting
- Space reclaimed by vacuum when needed

**Vacuum Algorithm:**

```python
# 1. Find chunks with has_deleted = 1
dirty_chunks = query_chunks_with_flag()

# 2. For each dirty chunk
for chunk_id in dirty_chunks:
    # Get all valid elements from index (by start_elem_idx, end_elem_idx)
    elements = query_elements_in_chunk(chunk_id)

    # Rebuild chunk with only valid data
    new_data = concat(valid_elements)
    new_chunk_size = len(new_data)

    replace_chunk(chunk_id, new_data, new_chunk_size)
    update_indices(elements)
    mark_chunk_clean(chunk_id)
```

**Benefits:**
- Reclaims space from deletions
- Rebuilds fragmented chunks
- Can be run periodically

#### ensure(name, dtype, chunk_bytes=None)

Get existing column or create if doesn't exist (idempotent).

```python
temps = cv.ensure("temperatures", "f64")
temps.append(23.5)

# Safe to call again (returns existing)
temps = cv.ensure("temperatures", "f64")
```

**Parameters:**
- **name** (str): Column name
- **dtype** (str): Data type (only used if creating)
- **chunk_bytes** (int, optional): Chunk size (only used if creating)

**Returns:** Column or VarSizeColumn

#### list_columns()

List all columns with metadata.

```python
cols = cv.list_columns()
for name, dtype, length in cols:
    print(f"{name}: {dtype} ({length} elements)")
```

**Returns:** List[(str, str, int)] - (name, dtype, length) tuples

#### delete_column(name)

Delete column and all its data.

```python
deleted = cv.delete_column("old_data")
if deleted:
    print("Column removed")
```

**Parameters:**
- **name** (str): Column name

**Returns:** bool - True if deleted, False if not found

**Warning:** This is permanent and cascades to all chunks!

### Cache Methods (v0.4.1)

Write-back cache for high-performance append operations.

#### enable_cache(cap_bytes=64MB, flush_threshold=16MB, flush_interval=None)

Enable cache for ALL columns in vault.

```python
# Enable with defaults (64MB capacity, 16MB threshold)
cv.enable_cache()

# Custom sizes
cv.enable_cache(cap_bytes=128*1024*1024, flush_threshold=32*1024*1024)

# With background daemon (flushes after 5 seconds idle)
cv.enable_cache(flush_interval=5.0)
```

**Parameters:**
- **cap_bytes** (int): Maximum cache size per column (default 64MB). When exceeded, auto-flushes.
- **flush_threshold** (int): Soft limit for auto-flush (default 16MB)
- **flush_interval** (float, optional): Enables daemon thread that auto-flushes after idle time

#### flush_cache()

Manually flush all column caches.

```python
bytes_flushed = cv.flush_cache()
```

**Returns:** int - Total bytes flushed across all columns

#### disable_cache()

Disable cache (auto-flushes first) and stops daemon thread.

```python
cv.disable_cache()
```

#### cache() context manager

Temporary cache enablement with auto-flush on exit.

```python
with cv.cache(cap_bytes=64<<20):
    col1.extend(data1)
    col2.extend(data2)
    # Auto-flushes here
```

#### lock_cache() context manager

Prevent daemon flushes during atomic operations.

```python
with cv.lock_cache():
    col1.append(value1)
    col2.append(value2)
    # Ensures both cached together before daemon flush
```

**Performance Impact:**
- **Cached append**: 10-100x faster for small operations
- **Cached extend**: 2-10x faster (reduces write amplification)

## Column (Fixed-Size Types)

List-like interface for fixed-size elements.

**Supported types:** `i64`, `f64`, `bytes:N`

### Indexing

```python
# Get element
value = col[0]
value = col[-1]        # Negative indexing supported

# Set element
col[0] = 42
col[-1] = 99

# Delete element (O(n) - shifts remaining)
del col[0]
```

**Raises:**
- `IndexError`: Index out of bounds
- `TypeError`: Invalid index type or value type

### Methods

#### append(value)

Append element to end. **O(1) amortized** - most efficient operation.

```python
col.append(42)
```

**Parameters:**
- **value**: Must match column dtype (int for i64, float for f64, bytes for bytes:N)

#### extend(values)

Append multiple elements efficiently.

```python
col.extend([1, 2, 3, 4, 5])
```

**Parameters:**
- **values** (list): List of values matching column dtype

**Complexity:** O(k) where k = len(values)

#### insert(idx, value)

Insert element at index. **O(n) - shifts elements after idx**.

```python
col.insert(0, value)  # Insert at beginning
col.insert(5, value)  # Insert at position 5
```

**Warning:** Slow for large columns. Use `append()` when possible.

#### clear()

Remove all elements.

```python
col.clear()
assert len(col) == 0
```

#### Cache Methods (v0.4.1)

Per-column cache control (same API as ColumnVault).

```python
# Enable cache for this column only
col.enable_cache(cap_bytes=64<<20, flush_threshold=16<<20)

# Context manager
with col.cache():
    for i in range(10000):
        col.append(i)  # 10-100x faster!

# Manual control
col.flush_cache()  # Returns bytes flushed
col.disable_cache()
```

**Note:** Cache auto-flushes before structural operations (insert, delete, clear).

### Query Methods

```python
# Length
count = len(col)

# Iteration
for value in col:
    print(value)

# Convert to list
all_values = list(col)

# Membership (slow - O(n) scan)
if 42 in col:
    print("Found")
```

## VarSizeColumn (Variable-Size Bytes)

List-like interface for variable-length byte strings.

**Type:** `bytes` (no size suffix)

### Differences from Column

**Supported:**
- ✅ `col[i]` get (O(1))
- ✅ `len(col)` (O(1))
- ✅ `append(value)` (O(1))
- ✅ `extend(values)` (O(k))
- ✅ Iteration (O(n))
- ✅ `clear()` (O(1))

**Not supported:**
- ❌ `col[i] = value` (can't resize in-place)
- ❌ `del col[i]` (would shift all data)
- ❌ `insert(i, value)` (same reason)

### Usage

```python
cv.create_column("messages", "bytes")  # Variable-size
msgs = cv["messages"]

# Append different-size strings
msgs.append(b"short")
msgs.append(b"this is a much longer message")
msgs.append(b"x")

# Access
print(msgs[0])   # b'short' (exact size, no padding)
print(msgs[1])   # b'this is a much longer message'
print(msgs[-1])  # b'x'

# Iterate
for msg in msgs:
    print(msg.decode())

# Cache support (v0.4.1) - same API as Column
with msgs.cache():
    msgs.extend(large_data_list)  # 2-10x faster
```

### How It Works

**Two internal columns:**
```python
# Creating "messages" (dtype="bytes") creates:
messages_data: Column[bytes:1]  # Packed bytes (all concatenated)
messages_idx: Column[i64]       # Prefix sum of byte offsets
```

**Storage layout:**
```
Data column: [s][h][o][r][t][l][o][n][g][e][r][x]
             └─────┘└──────────┘└┘
               "short"  "longer"  "x"

Index column: [5, 11, 12]  ← Cumulative byte offsets
```

**Access algorithm:**
```python
def __getitem__(self, i):
    if i == 0:
        start = 0
    else:
        start = index[i-1]
    end = index[i]
    return data[start:end]
```

**Space overhead:** 8 bytes per element (i64 offset)

## Data Types

### Fixed-Size Types

| Type | Size | Python Type | Packing | Example |
|------|------|-------------|---------|---------|
| `i64` | 8 bytes | int | Little-endian | `col.append(123)` |
| `f64` | 8 bytes | float | IEEE 754 double | `col.append(3.14)` |
| `bytes:N` | N bytes | bytes | Zero-padded | `col.append(b"hello")` → `b"hello\x00\x00\x00"` |

**Fixed-size pros:**
- Predictable storage (N elements = N × elem_size bytes)
- Simple indexing (byte_offset = index × elem_size)
- Can update elements in-place
- Can delete elements (shifts data)

**Fixed-size cons:**
- Wastes space if data varies in size (e.g., strings)
- Must know max size upfront for `bytes:N`

### Variable-Size Types

| Type | Size | Python Type | Storage Method | Example |
|------|------|-------------|----------------|---------|
| `bytes` | Variable | bytes | Prefix sum index | `col.append(b"any length")` |

**Variable-size pros:**
- No wasted space (stores exact bytes)
- Perfect for strings, JSON, variable-length data
- Still O(1) random access (via index)

**Variable-size cons:**
- 8-byte overhead per element (index)
- Can't update elements (size may change)
- Can't delete elements (would need index rebuild)

## Dynamic Chunk Growth

### How It Works

**Like C++ std::vector** - start small, grow exponentially:

1. **Create**: First chunk at `min_chunk_bytes` (default 128KB)
2. **Fill**: Append elements until chunk full
3. **Grow**: Double chunk size (128KB → 256KB → 512KB → ...)
4. **Cap**: Stop growing at `max_chunk_bytes` (default 16MB)
5. **Overflow**: Create new chunk starting at `min_chunk_bytes`

### Critical Invariant

**All chunks 0 to k-1 are ALWAYS at `max_chunk_bytes` (completely filled).**

Only the last chunk (k) can be smaller than `max_chunk_bytes`. This ensures:
- Simple byte-based addressing (byte_offset = chunk_idx × max_chunk_bytes + offset_in_chunk)
- Predictable performance (no complex chunk size tracking needed for reads)
- Cross-chunk elements handled correctly

### Example Growth

**For i64 column (8 bytes/element), min=128KB, max=16MB:**

```
Chunk 0 lifecycle:
- Append 1st element: Create chunk at 128KB (16,384 elem capacity)
- Append until full (16,384 elements): Chunk is full
- Append next element: Double to 256KB (32,768 elem capacity)
- Append until full (32,768 elements): Chunk is full
- Continue: 512KB → 1MB → 2MB → 4MB → 8MB → 16MB (max)
- At 16MB: Chunk holds 2,097,152 elements
- Append when full: Grow chunk 0 to max (16MB), create chunk 1 at 128KB

Chunk 1 lifecycle:
- Created: 128KB (starts growth cycle again)
- Growth repeats: 128KB → 256KB → ... → 16MB
```

**Key insight**: Chunk 0 reaches 16MB and is **completely filled** before chunk 1 is created.

### Performance Analysis

**Amortized O(1) append:**

When appending N elements:
- Total copies during growth: ~N (geometric series)
- Amortized cost per element: O(1)
- Same as Python list or C++ vector

**Space efficiency:**

| N elements | Fixed chunks (1MB each) | Dynamic chunks (128KB-16MB) |
|------------|------------------------|----------------------------|
| 10K (80KB) | 1MB (92% waste) | 128KB (38% waste) |
| 100K (800KB) | 1MB (20% waste) | 1MB (20% waste) |
| 1M (8MB) | 8MB (0% waste) | 8MB (0% waste) |
| 10M (80MB) | 80MB | 80MB |

**Key insight:** Dynamic growth wastes less space for small columns while maintaining good performance for large columns.

### Configuration

```python
# Small columns (reduce overhead)
cv = ColumnVault(
    "small.db",
    min_chunk_bytes=32*1024,    # 32KB start
    max_chunk_bytes=1*1024*1024 # 1MB max
)

# Large columns (fewer chunks)
cv = ColumnVault(
    "large.db",
    min_chunk_bytes=1*1024*1024,  # 1MB start
    max_chunk_bytes=128*1024*1024 # 128MB max
)
```

## Underlying Implementation

### Schema

```sql
CREATE TABLE col_meta (
    col_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    dtype TEXT NOT NULL,
    elem_size INTEGER NOT NULL,          -- Bytes per element
    length INTEGER NOT NULL,              -- Total elements
    chunk_bytes INTEGER NOT NULL,         -- (Deprecated)
    min_chunk_bytes INTEGER DEFAULT 131072,    -- 128KB
    max_chunk_bytes INTEGER DEFAULT 16777216   -- 16MB
);

CREATE TABLE col_chunks (
    col_id INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    data BLOB NOT NULL,                  -- Packed elements
    actual_size INTEGER NOT NULL,        -- Current chunk size
    PRIMARY KEY (col_id, chunk_idx),
    FOREIGN KEY (col_id) REFERENCES col_meta(col_id) ON DELETE CASCADE
);
```

### Indexing Algorithm

**For fixed-size columns:**

```python
# Given: index i, elem_size, max_chunk_bytes

# Step 1: Convert element index → byte offset
byte_offset = i * elem_size

# Step 2: Calculate chunk location (using max_chunk_bytes for addressing)
chunk_idx = byte_offset / max_chunk_bytes
offset_in_chunk = byte_offset % max_chunk_bytes

# Step 3: Read from chunk (may span multiple chunks!)
data = read_from_chunk(chunk_idx, offset_in_chunk, elem_size)

# Step 4: Unpack
value = struct.unpack("<q", data)[0]  # For i64
```

**Key invariant used**: All chunks 0..k-1 are at `max_chunk_bytes`, so simple division works!

**Rust handles:** Steps 2-3 (chunk addressing, BLOB access, cross-chunk reads)
**Python handles:** Steps 1, 4 (type awareness, packing/unpacking)

### Cross-Chunk Elements

**Elements can span chunk boundaries** when `max_chunk_bytes % elem_size != 0`.

Example with `elem_size=10`, `max_chunk_bytes=1024`:
```
Elements per chunk: 1024 / 10 = 102 (leaves 4 bytes remainder)

Chunk 0: Elements 0-101 (1020 bytes used, 4 bytes unused)
Chunk 1: Elements 102+ start here

Element 102 position:
- Byte offset: 102 × 10 = 1020
- Chunk: 1020 / 1024 = 0
- Offset in chunk: 1020 % 1024 = 1020
- **Crosses boundary**: Bytes 1020-1023 in chunk 0, bytes 1024-1029 in chunk 1
```

**Implementation**: `read_range` and `write_range` use byte-based addressing to automatically handle cross-chunk elements by:
1. Calculate byte range needed
2. Iterate over all chunks that contain any part of the range
3. Read/write the portion from each chunk
4. Assemble complete element from fragments

### Append Algorithm

**Python:**
```python
def append(self, value):
    packed = pack_i64(value)  # 8 bytes
    self._inner.append_raw(col_id, packed, elem_size=8, current_length)
```

**Rust (simplified):**
```rust
fn append_raw(data_bytes) {
    let remaining = data_bytes.len();
    let mut written = 0;

    while written < data_bytes.len() {
        // 1. Prepare chunk (grow/create as needed)
        let (chunk_idx, offset, capacity) = prepare_append_chunk(remaining);

        // 2. Write portion that fits in this chunk
        let to_write = min(remaining, capacity - offset);
        blob_write_at(chunk_idx, offset, data_bytes[written..written+to_write]);

        written += to_write;
        remaining -= to_write;
    }

    // 3. Update col_meta.length
}
```

**Growth decision tree for `prepare_append_chunk`:**
```
prepare_append_chunk(current_offset, remaining_bytes)
  ↓
Last chunk exists?
  No → Create chunk 0 with size = min(2^k × min_chunk, max_chunk, remaining)
       Return (chunk_idx=0, offset=0, capacity=chunk_size)
  Yes ↓
  ↓
Calculate space_left in current chunk
  ↓
space_left >= remaining?
  Yes → Return current chunk (has enough space)
  No ↓
  ↓
Try to double current chunk size
  actual_size × 2 (capped at max_chunk_bytes)
  ↓
After doubling, enough space?
  Yes → Return current chunk
  No ↓
  ↓
Grow current chunk to max_chunk_bytes
  ↓
After growing to max, has ANY space left?
  Yes → Return current chunk (fill remaining space first!)
  No ↓
  ↓
Current chunk completely full at max_chunk_bytes
  → Create new chunk with size = min(2^k × min_chunk, max_chunk, remaining)
  → Return new chunk (starts at offset 0)
```

**Critical behavior**: Always fill current chunk completely before creating next chunk!

## Performance Characteristics

### Complexity

| Operation | Fixed-Size | Variable-Size | Implementation |
|-----------|------------|---------------|----------------|
| `col[i]` get | O(log C) | O(1) | C = num chunks, find via byte offset |
| `col[i]` set | O(log C) | ❌ Not supported | Find chunk + write |
| `append()` | O(1) amortized | O(1) amortized | Rare chunk allocation/growth |
| `extend(k)` | O(k) | O(k) | Batch processing |
| `insert(i, v)` | O(n) | ❌ Not supported | Shift n-i elements |
| `del col[i]` | O(n) | ❌ Not supported | Shift n-i elements |
| Iteration | O(n) | O(n) | Chunked reads (batch 1000) |
| `len()` | O(1) | O(1) | Read from col_meta |
| `clear()` | O(1) | O(1) | Update col_meta.length |

**Key:**
- n = number of elements
- k = number of items in extend()
- C = number of chunks (typically log(n) for dynamic growth)

### Growth Cost Analysis

**For appending N elements with dynamic growth:**

```
Chunk growths: 128KB → 256KB → 512KB → 1MB → 2MB → 4MB → 8MB → 16MB

Total elements before max:
  16K + 32K + 64K + 128K + 256K + 512K + 1M + 2M ≈ 4M elements (for i64)

Copy cost:
  Growth 1: Copy 16K elements
  Growth 2: Copy 32K elements
  ...
  Growth 8: Copy 2M elements
  Total: ~4M copies for storing 2M elements → Amortized 2 copies/element
```

**Amortized append cost:** O(1) with factor ≈ 2

### Performance Optimizations

**1. Incremental BLOB I/O**

Append operations use SQLite's incremental BLOB API instead of reading entire chunks:

```rust
// BAD: Read entire 16MB chunk, modify 8 bytes, write back
let mut chunk_data = read_blob(chunk_idx);  // 16MB read
chunk_data[offset..offset+8] = new_data;
write_blob(chunk_idx, chunk_data);          // 16MB write

// GOOD: Write directly to chunk without reading
let blob = blob_open(chunk_idx, writable=true);
blob.write_at(new_data, offset);  // Just 8 bytes written
```

**Impact**: 10,000 appends = 80KB writes instead of 320GB I/O!

**2. Chunk Size Check Before Read**

Before growing a chunk, check `actual_size` column (integer) first:

```rust
// BAD: Always read blob to check size
let chunk_data = read_blob(chunk_idx);  // 16MB read
if chunk_data.len() >= max_chunk { return; }

// GOOD: Check actual_size column first
let actual_size = query("SELECT actual_size ...")[0];  // Integer read
if actual_size >= max_chunk { return; }  // Skip blob read!
```

**Impact**: Eliminates 16MB blob reads on every append after chunk reaches max.

**3. Exponential Growth on Large Appends**

When appending large data (e.g., 1MB at once), new chunks are sized appropriately:

```rust
fn calculate_new_chunk_size(remaining_bytes, min_chunk, max_chunk) {
    // Start with min, double until it fits remaining
    let mut size = min_chunk;
    while size < remaining && size < max_chunk {
        size *= 2;
    }
    return min(size, max_chunk);
}

// Examples:
// remaining = 100 bytes, min = 16 bytes, max = 1024 bytes
// → 16 < 100 → 32 < 100 → 64 < 100 → 128 >= 100 → size = 128 bytes

// remaining = 5000 bytes, min = 16 bytes, max = 1024 bytes
// → Keep doubling until 1024 (max) → size = 1024 bytes
```

**Impact**: Prevents creating thousands of tiny chunks when appending large data.

### Memory Usage

**Python-side:**
- Column objects: ~200 bytes each
- Cache: Minimal (length cached)

**Rust-side:**
- Minimal per-column overhead
- No data cached (read from SQLite)

**SQLite-side:**
- Page cache: Configurable (see KVault `cache_kb`)
- WAL buffer: Automatic

**Disk usage:**
- Column data: elem_size × length
- Index (varsize): 8 × length
- Chunk overhead: Negligible (metadata only)
- Wasted space: At most 50% of last chunk (on average 25%)

## Type Packing Details

### i64 (Signed Integer)

```python
# Pack (Python → bytes)
struct.pack("<q", value)  # Little-endian signed long long

# Unpack (bytes → Python)
struct.unpack("<q", data)[0]
```

**Range:** -2^63 to 2^63-1
**Storage:** 8 bytes, two's complement

### f64 (Float)

```python
# Pack
struct.pack("<d", value)  # Little-endian double

# Unpack
struct.unpack("<d", data)[0]
```

**Format:** IEEE 754 double precision
**Range:** ±1.7e308, 15-17 decimal digits precision
**Storage:** 8 bytes

### bytes:N (Fixed-Size Bytes)

```python
# Pack (pad with zeros)
value.ljust(N, b"\x00")

# Unpack
data[offset:offset+N]  # May contain trailing zeros
```

**Example:**
```python
col = cv.create_column("names", "bytes:20")
col.append(b"Alice")
# Stored as: b'Alice\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

name = col[0].rstrip(b'\x00')  # Remove padding
```

### bytes (Variable-Size)

**Storage strategy:**
- Data column: All bytes concatenated
- Index column: Cumulative byte offsets (prefix sum)

**Example:**
```python
col = cv.create_column("messages", "bytes")
col.append(b"hello")        # 5 bytes
col.append(b"world!")       # 6 bytes
col.append(b"x")            # 1 byte

# Stored as:
# messages_data: b"helloworld!x"  (12 bytes total)
# messages_idx:  [5, 11, 12]      (prefix sums)

# Retrieval:
col[0]  # data[0:5] = b"hello"
col[1]  # data[5:11] = b"world!"
col[2]  # data[11:12] = b"x"
```

## Limitations & Workarounds

### No Slicing

```python
# Not supported
subset = col[10:20]

# Workaround: Iterate
subset = [col[i] for i in range(10, 20)]

# Or: Read full list and slice
all_values = list(col)
subset = all_values[10:20]
```

### VarSize Element Updates

```python
col = cv.ensure("strings", "bytes")
col.append(b"hello")

# Can't do this (different size)
col[0] = b"world!"  # NotImplementedError

# Workaround 1: Clear and rebuild
values = list(col)
values[0] = b"world!"
col.clear()
col.extend(values)

# Workaround 2: Use fixed-size if updates needed
col = cv.create_column("strings", "bytes:100")  # Fixed
col[0] = b"world!"  # OK (padded to 100)
```

### Large Element Inserts

```python
# Slow (O(n) each)
for i in range(1000):
    col.insert(0, value)  # Prepend = shift everything

# Fast (O(1) each)
values = []
for i in range(1000):
    values.append(value)
col.extend(values)
```

### Multi-Column Transactions

```python
# Not atomic across columns
col1.append(1)
col2.append(2)  # If this fails, col1 change persists

# Workaround: Use SQLite transactions (advanced)
# Or: Design schema to minimize need
```

## Best Practices

### Hybrid Pattern: Columnar Metadata + KV Binaries

**For applications with many large binary files (images, videos, documents):**

```python
from kohakuvault import KVault, ColumnVault

kv = KVault("media_db.db")
cv = ColumnVault(kv)  # Share database

# Metadata in columnar storage (efficient for millions of entries)
cv.create_column("file_ids", "i64")
cv.create_column("filenames", "bytes")      # Variable-size strings
cv.create_column("file_sizes", "i64")
cv.create_column("content_types", "bytes:64")  # "image/jpeg", "video/mp4", etc.
cv.create_column("timestamps", "i64")

file_ids = cv["file_ids"]
filenames = cv["filenames"]
file_sizes = cv["file_sizes"]
content_types = cv["content_types"]
timestamps = cv["timestamps"]

# Ingest loop
for file_id, binary_data, filename, content_type in file_stream:
    # Fast O(1) append to columnar
    file_ids.append(file_id)
    filenames.append(filename)
    file_sizes.append(len(binary_data))
    content_types.append(content_type.encode())
    timestamps.append(int(time.time()))

    # Store binary in KV (streaming support for large files)
    kv[f"binary:{file_id}"] = binary_data

# Query metadata efficiently (without loading binaries!)
print(f"Total files: {len(file_ids)}")

# Filter by metadata
for i in range(len(file_ids)):
    if file_sizes[i] > 10 * 1024 * 1024:  # Files > 10MB
        print(f"Large file: {filenames[i].decode()} ({file_sizes[i]} bytes)")

        # Load binary only when needed
        if need_to_process(i):
            binary = kv[f"binary:{file_ids[i]}"]
            process(binary)

# Efficient iteration over specific content types
for i in range(len(file_ids)):
    ctype = content_types[i].rstrip(b'\x00').decode()
    if ctype.startswith("image/"):
        # Process only images
        image_data = kv[f"binary:{file_ids[i]}"]
        generate_thumbnail(image_data)
```

**Benefits of this pattern:**
- **Metadata queries** don't load binaries (fast filtering/scanning)
- **Columnar append** is O(1) amortized, handles millions of entries
- **KV streaming** handles multi-GB files without memory pressure
- **Single file** deployment (both in same SQLite database)
- **Flexible indexing** - add columns for new metadata without touching binaries
- **Partial loading** - query metadata, load binaries on demand

**Use this when:**
- Storing 1,000+ binary files
- Need to query/filter by metadata (size, type, date, etc.)
- Want to iterate over file list without loading all binaries
- Building media libraries, document stores, data lakes

**Don't use when:**
- Small number of binaries (< 100) - just use KV directly
- Always need full binary with metadata - no filtering benefit

### Choosing Data Types

**For integers:**
```python
# Small range (-2B to 2B): Use i64 (no i32 type yet)
cv.create_column("counters", "i64")

# Large integers: Use bytes (pack manually)
import struct
col = cv.create_column("bigint", "bytes:16")
col.append(struct.pack("<QQ", low_64bits, high_64bits))
```

**For floats:**
```python
# Precision needed: Use f64
cv.create_column("prices", "f64")

# Lower precision OK: Use bytes:4 + manual packing
col = cv.create_column("approx", "bytes:4")
col.append(struct.pack("<f", value))  # 32-bit float
```

**For strings:**
```python
# Known max length: Use bytes:N (faster)
cv.create_column("usernames", "bytes:50")

# Unknown length: Use bytes (variable)
cv.create_column("descriptions", "bytes")
```

### Choosing Chunk Sizes

**General rule:**
- **min_chunk_bytes**: ~1000-10000 elements worth
- **max_chunk_bytes**: Based on RAM and access pattern

**Examples:**
```python
# Tiny columns (< 1K elements expected)
min=16*1024,  max=256*1024    # 16KB-256KB

# Small columns (1K-100K elements)
min=128*1024, max=16*1024*1024  # 128KB-16MB (default)

# Large columns (millions of elements)
min=1*1024*1024, max=128*1024*1024  # 1MB-128MB
```

## Examples

See `examples/columnar_demo.py` for complete examples:
- Time-series sensor data
- Application logs with variable messages
- ML feature vectors
- User activity tracking
- Chunk growth demonstration

## See Also

- `docs/arch.md` - Overall architecture
- `docs/kv.md` - Key-value API
- `docs/COLUMNAR_GUIDE.md` - Columnar usage guide
- `examples/columnar_demo.py` - Working examples
