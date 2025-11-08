# KohakuVault

SQLite-backed storage with Rust performance. Works like Python dict/list but persisted and fast.

## Installation

```bash
pip install kohakuvault
```

## KVault - Works Like a Dict

```python
from kohakuvault import KVault
import numpy as np

kv = KVault("app.db")

# Just use it like a dict - store anything
kv["user:123"] = {"name": "Alice", "email": "alice@example.com"}
kv["embeddings"] = np.random.randn(768).astype(np.float32)
kv["settings"] = {"theme": "dark", "notifications": True}
kv["scores"] = [95.5, 87.3, 92.1]

# Get back actual Python objects
user = kv["user:123"]  # dict
embeddings = kv["embeddings"]  # numpy array
settings = kv["settings"]  # dict

# Standard dict operations
del kv["user:123"]
if "settings" in kv:
    print("Settings exist")

# Bulk operations with cache for speed
with kv.cache():
    for i in range(100000):
        kv[f"item:{i}"] = {"id": i, "data": f"value_{i}"}
```

**That's it.** No manual pickle/msgpack/json. Just works.

## ColumnVault - Works Like Typed Lists

```python
from kohakuvault import ColumnVault

cv = ColumnVault("data.db")

# Create typed columns
user_ids = cv.create_column("user_ids", "i64")
scores = cv.create_column("scores", "f64")
comments = cv.create_column("comments", "str:utf8")
metadata = cv.create_column("metadata", "msgpack")

# Use like lists
user_ids.append(1)
user_ids.extend([2, 3, 4, 5])

scores.append(95.5)
scores.extend([87.3, 92.1, 88.0])

# Indexing and slicing
first_score = scores[0]
batch = scores[10:20]  # Fast batch read

# Updates
scores[5] = 99.0
scores[10:15] = [100.0, 101.0, 102.0, 103.0, 104.0]

# Iterate
for score in scores[:10]:
    print(score)
```

**Built-in types**: `i64`, `f64`, `str:utf8`, `bytes`, `msgpack`, `cbor`

## Vector Storage

Store arrays/tensors efficiently:

```python
# Embeddings (768-dim vectors)
embeddings = cv.create_column("embeddings", "vec:f32:768")
embeddings.append(np.random.randn(768).astype(np.float32))
embeddings.extend([model.encode(text) for text in documents])

# Images (28x28 grayscale)
images = cv.create_column("mnist", "vec:u8:28:28")
images.append(np.random.randint(0, 256, (28, 28), dtype=np.uint8))

# RGB images (224x224x3)
photos = cv.create_column("photos", "vec:u8:3:224:224")

# Matrices
matrices = cv.create_column("correlations", "vec:f64:100:100")
```

**Vector types**: `vec:f32:768` (fixed), `vec:i64:10:20` (2D), `vec:u8:3:224:224` (3D), `vec:f32` (arbitrary)

## Vector Search

Find similar vectors:

```python
from kohakuvault import VectorKVault

# Create search index
search = VectorKVault("search.db", dimensions=384, metric="cosine")

# Add vectors
for doc, embedding in zip(documents, embeddings):
    search.insert(embedding, doc.encode())

# Search for similar
results = search.search(query_embedding, k=10)
for id, distance, doc in results:
    print(f"{distance:.3f}: {doc.decode()}")

# Get single closest
closest = search.get(query_embedding)
```

**Metrics**: `cosine` (text), `l2` (images), `l1`, `hamming`

## Performance Tips

### Use Cache for Bulk Writes

```python
# 10-100x faster with cache
with kv.cache():
    for i in range(100000):
        kv[f"key:{i}"] = data

# Or enable cache
kv.enable_cache()
# ... write operations ...
kv.flush_cache()
```

### Batch Operations Beat Loops

```python
# Slow: loop
for i in range(1000):
    column.append(value)

# Fast: bulk extend (450x faster!)
column.extend([value] * 1000)

# Slow: loop read
for i in range(100):
    item = column[i]

# Fast: slice read (200x faster!)
batch = column[0:100]
```

### Use DataPacker for Custom Serialization

```python
from kohakuvault import DataPacker

# Primitives
packer_i64 = DataPacker("i64")
packed = packer_i64.pack(42)

# MessagePack for dicts/lists
packer_msg = DataPacker("msgpack")
packed = packer_msg.pack({"key": "value"})

# Bulk operations (3-35x faster!)
values = list(range(100000))
packed = packer_i64.pack_many(values)
unpacked = packer_i64.unpack_many(packed, count=100000)
```

## Real-World Example

```python
from kohakuvault import KVault, ColumnVault, VectorKVault
import numpy as np

# All components share one database file
db = "app.db"

# 1. Store documents
kv = KVault(db, table="docs")
for doc_id, content in documents:
    kv[doc_id] = content  # Auto-packs to MessagePack or keeps bytes

# 2. Store metadata in columns
cv = ColumnVault(db)
ids = cv.create_column("doc_ids", "i64")
titles = cv.create_column("titles", "str:utf8")
vectors = cv.create_column("embeddings", "vec:f32:384")

for i, (title, embedding) in enumerate(zip(titles_list, embeddings_list)):
    ids.append(i)
    titles.append(title)
    vectors.append(embedding)

# 3. Build search index
search = VectorKVault(db, table="search", dimensions=384, metric="cosine")
for i, embedding in enumerate(embeddings_list):
    search.insert(embedding, str(i).encode())

# 4. Search and retrieve
query = model.encode("search query")
results = search.search(query, k=5)

for rank, (vec_id, distance, doc_idx_bytes) in enumerate(results, 1):
    idx = int(doc_idx_bytes.decode())
    title = titles[idx]
    content = kv[f"doc:{idx}"]
    print(f"{rank}. {title} (similarity: {1-distance:.2f})")
```

## Performance

**M1 Max benchmarks**:
- KVault: 24K writes/s, 63K reads/s (with cache)
- Column extend: 12.5M ops/s (i64)
- Column slice: 2.3M slices/s (100 items)
- Vector unpack: 35x faster bulk vs loop
- MessagePack: 42% smaller than JSON

See `examples/benchmark.py`

## Features

- **Auto-packing**: Store numpy, dict, list, int, float, str automatically
- **Vector search**: k-NN similarity search (cosine, L2, L1, hamming)
- **Vector storage**: Arrays/tensors in columns (1-byte overhead)
- **Caching**: Write-back cache with auto-flush
- **Batch operations**: Slice read/write, pack_many/unpack_many
- **Single file**: All components share one SQLite database
- **Ordered containers**: CSBTree (B+Tree), SkipList (lock-free)
- **Type wrappers**: `MsgPack(data)`, `Json(data)`, `Cbor(data)`, `Pickle(data)`

## Storage Types

**KVault** (dict-like):
- Any Python object (auto-packed)
- Works with cache for performance

**ColumnVault** (list-like):
- `i64`, `f64` - Integers, floats
- `str:utf8` - Strings
- `bytes`, `bytes:N` - Raw bytes
- `msgpack`, `cbor` - Structured data
- `vec:f32:768`, `vec:u8:28:28` - Arrays/vectors

**VectorKVault** (search):
- k-NN similarity search
- Multiple metrics
- Numpy arrays

## Development

```bash
pip install -e .[dev]
pytest
cargo test
cargo clippy
```

## Documentation

- `examples/basic_usage.py` - Quick examples
- `examples/all_usage.py` - Complete feature tour
- `docs/` - Detailed guides

## License

Apache 2.0
