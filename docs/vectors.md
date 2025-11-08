# Vector Storage and Search

KohakuVault 0.7.0 adds comprehensive vector storage and similarity search capabilities.

## Vector Storage (ColumnVault + DataPacker)

### Fixed-Shape Vectors (Recommended)

```python
from kohakuvault import ColumnVault
import numpy as np

cv = ColumnVault("vectors.db")

# Text embeddings (768-dim)
embeddings = cv.create_column("bert_embeddings", "vec:f32:768")
embeddings.append(np.random.randn(768).astype(np.float32))
embeddings.extend([np.random.randn(768).astype(np.float32) for _ in range(1000)])

# Images (28x28 grayscale)
images = cv.create_column("mnist", "vec:u8:28:28")

# RGB images (224x224x3)
photos = cv.create_column("imagenet", "vec:u8:3:224:224")

# Matrices (10x20)
matrices = cv.create_column("correlations", "vec:f64:10:20")
```

**Overhead**: 1 byte per vector (type byte)

### Arbitrary-Shape Vectors

```python
# Variable shapes
generic_vectors = cv.create_column("generic", "vec:f32")

# Store different shapes
generic_vectors.append(np.random.randn(100).astype(np.float32))      # 1D
generic_vectors.append(np.random.randn(10, 20).astype(np.float32))   # 2D
generic_vectors.append(np.random.randn(5, 5, 5).astype(np.float32))  # 3D
```

**Overhead**: 2 + ndim*4 bytes (shape metadata included)

### Supported Element Types

- `f32`, `f64` - Floating point
- `i32`, `i64`, `i8`, `i16` - Signed integers
- `u8`, `u16`, `u32`, `u64` - Unsigned integers

### Bulk Operations (Optimized)

```python
from kohakuvault import DataPacker

packer = DataPacker("vec:f32:768")

# 35x faster than loop!
vectors = [np.random.randn(768).astype(np.float32) for _ in range(10000)]
packed = packer.pack_many(vectors)
unpacked = packer.unpack_many(packed, count=10000)  # Returns list of numpy arrays
```

## Vector Similarity Search (VectorKVault)

### Basic Usage

```python
from kohakuvault import VectorKVault
import numpy as np

# Create search index
vkv = VectorKVault("search.db", dimensions=384, metric="cosine")

# Index documents
for doc_id, (text, embedding) in enumerate(zip(documents, embeddings)):
    vkv.insert(embedding, text.encode())

# k-NN search
query_embedding = model.encode("search query")
results = vkv.search(query_embedding, k=10)

for rank, (id, distance, doc_bytes) in enumerate(results, 1):
    print(f"{rank}. Distance={distance:.4f} | {doc_bytes.decode()}")

# Single closest match
closest_doc = vkv.get(query_embedding)
```

### Similarity Metrics

- **cosine**: Cosine similarity (good for text embeddings)
- **l2**: Euclidean distance (good for images)
- **l1**: Manhattan distance
- **hamming**: Hamming distance (for binary vectors)

```python
# Different metrics for different use cases
text_search = VectorKVault("text.db", dimensions=384, metric="cosine")
image_search = VectorKVault("images.db", dimensions=512, metric="l2")
```

### CRUD Operations

```python
# Insert
doc_id = vkv.insert(vector, value)

# Search k-nearest
results = vkv.search(query_vector, k=10, metric=None)

# Get single closest
closest = vkv.get(query_vector)

# Get by ID (returns numpy array for vector)
vector, value = vkv.get_by_id(doc_id)

# Update
vkv.update(doc_id, vector=new_embedding)
vkv.update(doc_id, value=new_content)

# Delete
vkv.delete(doc_id)

# Check existence
if doc_id in vkv:
    print("Exists!")
```

### Performance Characteristics

- **Small datasets** (1K-10K vectors): <1ms per query
- **Medium datasets** (10K-100K vectors): <10ms per query
- **SIMD acceleration**: Automatic on x86 (AVX) and ARM (NEON)
- **Brute-force scan**: No ANN indexes yet (sqlite-vec v0.1.6)

## Auto-Packing Integration

### With KVault

```python
from kohakuvault import KVault

kv = KVault("data.db")  # Auto-pack enabled by default

# Store vectors directly in KVault
kv["model_weights"] = np.random.randn(1000, 768).astype(np.float32)

# Retrieve as numpy array
weights = kv["model_weights"]  # np.ndarray, not bytes!
```

### With ColumnVault

```python
cv = ColumnVault("data.db")

# Vector columns
embeddings = cv.create_column("embeddings", "vec:f32:768")

# Mixed columns in same database
ids = cv.create_column("ids", "i64")
metadata = cv.create_column("metadata", "msgpack")

# All stored efficiently
for i in range(1000):
    ids.append(i)
    embeddings.append(np.random.randn(768).astype(np.float32))
    metadata.append({"timestamp": i, "source": "model_v1"})
```

## Combined: Semantic Search System

```python
from kohakuvault import KVault, ColumnVault, VectorKVault

# All components share one database file!
db = "semantic_search.db"

# 1. Store full documents in KVault
kv = KVault(db, table="documents")
for doc_id, doc_text in documents:
    kv[doc_id] = doc_text.encode()

# 2. Store metadata in ColumnVault
cv = ColumnVault(db)
titles = cv.create_column("titles", "str:utf8")
timestamps = cv.create_column("timestamps", "i64")

# 3. Build search index with VectorKVault
vkv = VectorKVault(db, table="search_index", dimensions=384, metric="cosine")
for doc_id, embedding in zip(doc_ids, embeddings):
    vkv.insert(embedding, doc_id.encode())

# 4. End-to-end search
query_embedding = model.encode("machine learning")
results = vkv.search(query_embedding, k=5)

for id, distance, doc_id_bytes in results:
    doc_id = doc_id_bytes.decode()
    full_doc = kv[doc_id]
    title = titles[int(doc_id.split(':')[1])]
    print(f"Distance={distance:.4f} | {title} | {full_doc.decode()[:100]}...")
```

## Binary Format

### Vector Format

**Fixed-shape**: `|type(1)|data...|`
- Example: `vec:f32:768` = 1 + 768*4 = 3073 bytes
- Minimal overhead: 0.19%

**Arbitrary-shape**: `|type(1)|ndim(1)|shape(ndim*4)|data...|`
- Example: 2D array [10, 20] = 1 + 1 + 8 + 800 = 810 bytes
- Overhead: 10 bytes for metadata

### Auto-Pack Header

When auto-packing is used with non-raw data:

`|magic_k(2)|version(1)|encoding(1)|flags(1)|reserved(3)|magic_q(2)|data...|`

- **magic_k**: `0x89, 0x4B`
- **magic_q**: `0x56, 0x4B` (validation)
- **encoding**: Raw(0x00), Pickle(0x01), DataPacker(0x02), Json(0x03), MessagePack(0x04), Cbor(0x05)

**Smart behavior**: Raw bytes (media files) get NO header, preserving external tool compatibility.

## See Also

- `examples/basic_usage.py` - Auto-packing examples
- `examples/all_usage.py` - Comprehensive feature demos
- `examples/vector_search_numpy.py` - Vector search walkthrough
- `docs/datapacker.md` - DataPacker details
- `docs/kv.md` - KVault details
- `docs/col.md` - ColumnVault details
