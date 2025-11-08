# Vector Storage & Search

KohakuVault 0.7 ships two complementary vector features:

1. `vec:*` dtypes inside `ColumnVault` for storing dense arrays/tensors alongside other columns.
2. `VectorKVault`, a sqlite-vec powered similarity search engine for k-NN queries.

Both live in the same SQLite file so you can ingest once and fan out to analytics, retrieval, and metadata workflows.

## Vector Columns (`ColumnVault`)

### Fixed-Shape Vectors

Use fixed-shape dtypes when every vector shares the same dimensions:

```python
from kohakuvault import ColumnVault
import numpy as np

cv = ColumnVault("vectors.db")
embeddings = cv.create_column("bert", "vec:f32:768")
embeddings.extend(np.random.randn(10_000, 768).astype(np.float32))
```

Overhead: 1 byte per vector (type byte). Ideal for embeddings, image tensors, and matrices with stable shapes (`vec:u8:3:224:224`).

### Arbitrary-Shape Vectors

```python
generic = cv.create_column("generic", "vec:f32")
generic.append(np.random.randn(100).astype(np.float32))
generic.append(np.random.randn(10, 20).astype(np.float32))
```

Overhead: `2 + ndim * 4` bytes (stores ndim and each dimension). Use when shapes vary but you still want zero-copy serialization.

### Supported Element Types

`f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` – matching `DataPacker`’s `ElementType` enum. Mixed precision? Convert before writing.

### Batch Operations with DataPacker

```python
from kohakuvault import DataPacker

packer = DataPacker("vec:f32:768")
vectors = [np.random.randn(768).astype(np.float32) for _ in range(10_000)]
buffer = packer.pack_many(vectors)
unpacked = packer.unpack_many(buffer, count=len(vectors))
```

Use this when you need to preprocess vectors before inserting or when marshaling data over the network. Arbitrary-shape vectors use `offsets=` instead of `count=`.

## VectorKVault (Similarity Search)

`VectorKVault` wraps sqlite-vec’s `vec0` virtual table and stores arbitrary values in a side table. It’s perfect for brute-force k-NN on tens of thousands of vectors.

```python
from kohakuvault import VectorKVault
import numpy as np

vkv = VectorKVault("search.db", table="search", dimensions=384, metric="cosine", vector_type="f32")

vector = np.random.randn(384).astype(np.float32)
doc_id = vkv.insert(vector, b"document payload")

results = vkv.search(vector, k=10)
for row_id, distance, payload in results:
    print(row_id, distance, payload[:32])

closest = vkv.get(vector)
vector2, payload2 = vkv.get_by_id(doc_id)
vkv.update(doc_id, vector=new_vector)
vkv.delete(doc_id)
```

### Configuration

| Parameter | Meaning |
|-----------|---------|
| `dimensions` | Length of every vector you insert (enforced). |
| `metric` | `"cosine"`, `"l2"`, `"l1"`, or `"hamming"`. Per-query overrides allowed via `search(..., metric="l2")`. |
| `vector_type` | `"f32"`, `"int8"`, or `"bit"` depending on sqlite-vec support. Defaults to `f32`. |
| `table` | Logical table name; both the vec0 virtual table and blob table derive from this. |

### CRUD & Introspection

| Method | Description |
|--------|-------------|
| `insert(vector, value, metadata=None)` | Inserts vector/value and returns rowid. `metadata` is reserved for future use. |
| `search(query, k=10, metric=None)` | Returns list of `(row_id, distance, value)` sorted by distance. |
| `get(query, metric=None)` | Shortcut for `k=1` that returns the value bytes. Raises `NotFound` if empty. |
| `get_by_id(row_id)` | Returns `(numpy_vector, value)` so you can inspect stored vectors. |
| `update(row_id, vector=None, value=None)` | Mutate stored vectors/values independently. |
| `delete(row_id)` | Remove entry. |
| `exists(row_id)` | Boolean check. |
| `count()` | Number of rows inside the vec0 table. |
| `info()` | Dict with table, dimensions, metric, vector type, and count. |

### Performance Characteristics

- sqlite-vec 0.1.6 performs brute-force scans with SIMD acceleration (AVX on x86, NEON on Apple Silicon). Expect <1 ms for 1K rows, <10 ms for 100K rows on modern hardware.
- Because scans touch the entire table, keep WALs short (`checkpoint()` regularly) and VACUUM after bulk deletes.
- Use `vector_type="int8"` plus quantized vectors when you need smaller storage or faster Hamming metrics.

## Putting It Together

```python
from kohakuvault import KVault, ColumnVault, VectorKVault
import numpy as np

db = "semantic.db"
kv = KVault(db, table="documents")
cv = ColumnVault(kv)
emb = cv.create_column("embeddings", "vec:f32:384")
titles = cv.create_column("titles", "str:utf8")

for idx, (title, text, embedding) in enumerate(dataset):
    kv[f"doc:{idx}"] = text.encode()
    titles.append(title)
    emb.append(embedding.astype(np.float32))

search = VectorKVault(db, table="search_index", dimensions=384, metric="cosine")
for idx, embedding in enumerate(emb[-len(dataset):]):
    search.insert(embedding, str(idx).encode())

query = model.encode("machine learning").astype(np.float32)
for rank, (row_id, dist, doc_idx_bytes) in enumerate(search.search(query, k=5), 1):
    doc_idx = int(doc_idx_bytes)
    print(rank, dist, titles[doc_idx], kv[f"doc:{doc_idx}"][:80])
```

Vector storage, metadata, and similarity search all share `semantic.db`. No extra services required.

For more on dtype grammar and caching, see [ColumnVault Guide](columnvault.md). For auto-pack details (storing vectors directly in `KVault`), see [KVault Guide](kvault.md).
