# TextVault – Full-Text Search with BM25 Ranking

KohakuVault 0.8 introduces `TextVault`, a full-text search storage built on SQLite's FTS5 extension. It combines the convenience of `KVault`'s auto-packing with powerful BM25-ranked search, making it ideal for RAG (Retrieval-Augmented Generation) pipelines and document retrieval systems.

## Quick Start

```python
from kohakuvault import TextVault

# Create a single-column text vault
tv = TextVault("documents.db")

# Insert documents with any Python value
doc_id = tv.insert("Machine learning is a subset of AI", {"category": "tech", "importance": 5})

# Search with BM25 ranking
results = tv.search("machine learning", k=10)
for doc_id, score, value in results:
    print(f"ID {doc_id}: score={score:.4f}, value={value}")
```

## Features

- **BM25 ranking** – Industry-standard relevance scoring via FTS5
- **Safe query escaping** – Special characters (?, +, @, etc.) handled automatically
- **Multi-column schemas** – Index title, body, tags separately
- **Auto-packing** – Store any Python object (dict, list, numpy) as values
- **Dict-like interface** – `tv["key"] = value` for simple key-value usage
- **Snippet highlighting** – Get highlighted matches in context

## Single-Column vs Multi-Column

### Single Column (Default)

Best for simple document storage:

```python
tv = TextVault("data.db")  # defaults to columns=["content"]
tv.insert("Hello world", {"meta": "data"})
value = tv["Hello world"]  # exact match lookup
```

### Multi-Column

Best for structured documents with separate searchable fields:

```python
tv = TextVault("data.db", columns=["title", "body", "tags"])
tv.insert(
    {"title": "Introduction to ML", "body": "Machine learning is...", "tags": "ml ai tutorial"},
    {"author": "John", "date": "2024-01-01"}
)

# Search specific columns
results = tv.search("introduction", column="title")
results = tv.search("tutorial", column="tags")
```

## Search Methods

### Basic Search

Returns `(id, bm25_score, value)` tuples sorted by relevance (higher = better):

```python
results = tv.search("hello world", k=10)
for doc_id, score, value in results:
    print(f"Doc {doc_id}: {score:.4f}")
```

### Search with Text Content

Returns the indexed text alongside results:

```python
results = tv.search_with_text("machine learning", k=5)
for doc_id, score, text, value in results:
    print(f"Text: {text}")
    print(f"Value: {value}")
```

### Search with Snippets

Get highlighted context around matches:

```python
results = tv.search_with_snippets(
    "machine learning",
    k=10,
    snippet_tokens=15,          # words around match
    highlight_start="<b>",      # custom highlight markers
    highlight_end="</b>"
)
for doc_id, score, snippet, value in results:
    print(f"Snippet: {snippet}")  # "...is a <b>machine learning</b> algorithm that..."
```

## Query Syntax

### Safe Mode (Default)

By default, `escape=True` wraps your query for safe literal matching. This handles FTS5 special characters automatically:

```python
tv.search("What is this?")      # Works with ?
tv.search("C++ programming")    # Works with +
tv.search("test@email.com")     # Works with @
tv.search('He said "hello"')    # Works with quotes
```

### Raw FTS5 Syntax

Set `escape=False` to use FTS5 query operators:

```python
tv.search("hello AND world", escape=False)     # Both terms required
tv.search("python OR java", escape=False)      # Either term
tv.search("hello NOT goodbye", escape=False)   # Exclusion
tv.search("mach*", escape=False)               # Prefix matching
tv.search("NEAR(hello world, 5)", escape=False) # Proximity search
```

## CRUD Operations

### Insert

```python
# Single column
doc_id = tv.insert("document text", {"key": "value"})

# Multi-column
doc_id = tv.insert(
    {"title": "Hello", "body": "World", "tags": "greeting"},
    {"created": "2024-01-01"}
)
```

### Get by ID

```python
text, value = tv.get_by_id(doc_id)
# Single column: text is string
# Multi-column: text is dict {"title": "...", "body": "...", "tags": "..."}
```

### Get by Exact Key (Single-Column Only)

```python
value = tv.get("exact document text")
# or dict-style:
value = tv["exact document text"]
```

### Update

```python
# Update text only
tv.update(doc_id, texts="new document text")

# Update value only
tv.update(doc_id, value={"new": "metadata"})

# Update both
tv.update(doc_id, texts="new text", value={"new": "value"})
```

### Delete

```python
tv.delete(doc_id)
# or for single-column with exact match:
del tv["exact document text"]
```

## Auto-Packing

TextVault uses the same auto-packing system as KVault and VectorKVault:

```python
# Store any Python object as values
tv.insert("text", {"dict": "value"})           # MessagePack
tv.insert("text", [1, 2, 3])                   # MessagePack
tv.insert("text", np.array([1, 2, 3]))         # DataPacker vec:*
tv.insert("text", 42)                          # DataPacker i64
tv.insert("text", b"raw bytes")               # Raw (no header)

# Values are automatically decoded on retrieval
text, value = tv.get_by_id(doc_id)
print(type(value))  # <class 'dict'>, <class 'list'>, <class 'numpy.ndarray'>, etc.
```

### Managing Auto-Pack

```python
# Check status
tv.auto_pack_enabled()  # True by default

# Disable for bytes-only mode
tv.disable_auto_pack()

# Re-enable
tv.enable_auto_pack(use_pickle=True)  # use_pickle allows custom objects
```

## Utility Methods

```python
len(tv)                    # Document count
tv.count()                 # Same as len()
tv.count_matches("query")  # Count matching documents
tv.exists(doc_id)          # Check if ID exists
doc_id in tv               # Same as exists()
tv.info()                  # {"table": "...", "columns": [...], "count": N}
tv.columns                 # ["content"] or ["title", "body", "tags"]
tv.keys(limit=100, offset=0)  # Paginated list of rowids
tv.clear()                 # Delete all documents
```

## RAG Pipeline Example

```python
from kohakuvault import TextVault, VectorKVault
import numpy as np

# Documents with text search
docs = TextVault("rag.db", table="documents")

# Embeddings with vector search
vecs = VectorKVault("rag.db", table="embeddings", dimensions=384, metric="cosine")

# Index documents
for idx, doc in enumerate(documents):
    # Store full document with metadata
    doc_id = docs.insert(doc["text"], {"source": doc["source"], "date": doc["date"]})

    # Store embedding linked to doc_id
    embedding = model.encode(doc["text"])
    vecs.insert(embedding.astype(np.float32), str(doc_id).encode())

# Hybrid search: combine BM25 + vector similarity
def hybrid_search(query: str, k: int = 10, alpha: float = 0.5):
    # BM25 search
    bm25_results = {doc_id: score for doc_id, score, _ in docs.search(query, k=k*2)}

    # Vector search
    query_vec = model.encode(query).astype(np.float32)
    vec_results = {int(doc_id): 1/(1+dist) for _, dist, doc_id in vecs.search(query_vec, k=k*2)}

    # Combine scores
    all_ids = set(bm25_results) | set(vec_results)
    combined = []
    for doc_id in all_ids:
        bm25_score = bm25_results.get(doc_id, 0)
        vec_score = vec_results.get(doc_id, 0)
        combined.append((doc_id, alpha * bm25_score + (1-alpha) * vec_score))

    return sorted(combined, key=lambda x: x[1], reverse=True)[:k]
```

## Performance Notes

- FTS5 uses an inverted index, so searches are fast even with millions of documents
- BM25 scoring is computed at query time with minimal overhead
- The separate values table keeps the FTS5 index lean
- Use `escape=True` (default) unless you need FTS5 operators
- For very large result sets, use `k` to limit results rather than post-filtering

## Schema Design

TextVault creates two tables:

1. **FTS5 virtual table** (`{table}`) – Stores indexed text columns + `value_ref`
2. **Values table** (`{table}_values`) – Stores blob values separately

This separation keeps the FTS5 index efficient while allowing arbitrary-size values.

```sql
-- What TextVault creates:
CREATE VIRTUAL TABLE text_vault USING fts5(content, value_ref UNINDEXED);
CREATE TABLE text_vault_values (id INTEGER PRIMARY KEY, value BLOB);
```

## See Also

- [KVault Guide](kvault.md) – Key-value storage with auto-packing
- [Vector Storage & Search](vectors.md) – VectorKVault for similarity search
- [DataPacker Reference](datapacker.md) – How auto-pack serialization works
