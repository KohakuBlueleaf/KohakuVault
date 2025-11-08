"""
KohakuVault Basic Usage - Simple Examples

Auto-packing is ENABLED BY DEFAULT - you can store any Python object!
- numpy arrays, dicts, lists, ints, floats, strings → automatic!
- bytes → stored as raw (media file compatible)

No manual serialization needed!
"""

import numpy as np

from kohakuvault import ColumnVault, KVault, VectorKVault


def example_1_kvault_auto_pack():
    """Example 1: KVault with auto-packing (DEFAULT - NO SETUP NEEDED!)."""
    print("=" * 80)
    print("Example 1: KVault Auto-Packing (Default Behavior)")
    print("=" * 80)

    kv = KVault(":memory:")

    # Auto-packing is enabled by default - just store any Python object!
    print("Storing different types (automatic serialization)...")

    # Numpy arrays → DataPacker vec:*
    kv["embedding"] = np.random.randn(128).astype(np.float32)

    # Dicts → MessagePack (efficient!)
    kv["config"] = {"timeout": 30, "retries": 3, "enabled": True}

    # Lists → MessagePack
    kv["scores"] = [95.5, 87.3, 92.1, 88.0, 91.5]

    # Primitives → DataPacker i64/f64
    kv["count"] = 42
    kv["score"] = 95.5

    # Strings → DataPacker str:utf8
    kv["name"] = "Alice"

    # Bytes → Raw (NO header, media file compatible!)
    kv["image.jpg"] = b"\xff\xd8\xff\xe0" + b"fake jpeg" * 100

    # Retrieve - automatically decoded!
    print("\nRetrieving (automatic deserialization)...")
    embedding = kv["embedding"]
    config = kv["config"]
    scores = kv["scores"]
    count = kv["count"]
    score = kv["score"]
    name = kv["name"]
    image = kv["image.jpg"]

    print(f"  embedding: type={type(embedding).__name__}, shape={embedding.shape}")
    print(f"  config: {config}")
    print(f"  scores: {scores}")
    print(f"  count: {count} (type: {type(count).__name__})")
    print(f"  score: {score} (type: {type(score).__name__})")
    print(f"  name: {name} (type: {type(name).__name__})")
    print(f"  image: {len(image)} bytes (type: {type(image).__name__})")

    print("\n✓ Auto-packing complete - no manual serialization needed!\n")
    kv.close()


def example_2_kvault_cache():
    """Example 2: Using cache for performance."""
    print("=" * 80)
    print("Example 2: KVault with Cache (Performance)")
    print("=" * 80)

    kv = KVault(":memory:")

    # Use cache context manager for bulk writes
    print("Writing 1000 items with cache...")
    with kv.cache():
        for i in range(1000):
            # Auto-packing works with cache!
            kv[f"item:{i}"] = {
                "id": i,
                "value": i * 1.5,
                "tags": [f"tag_{i % 5}", f"tag_{i % 7}"],
            }

    # Cache automatically flushed on exit
    print(f"Total items: {len(kv)}")

    # Retrieve a few
    print(f"\nitem:0 = {kv['item:0']}")
    print(f"item:999 = {kv['item:999']}")

    print("\n✓ Cache example complete\n")
    kv.close()


def example_3_column_vault_vectors():
    """Example 3: ColumnVault for structured vector storage."""
    print("=" * 80)
    print("Example 3: ColumnVault - Vector Columns")
    print("=" * 80)

    cv = ColumnVault(":memory:")

    # Create columns
    ids = cv.create_column("ids", "i64")
    embeddings = cv.create_column("embeddings", "vec:f32:768")
    labels = cv.create_column("labels", "str:utf8")

    print("Adding 100 records...")
    for i in range(100):
        ids.append(i)
        embeddings.append(np.random.randn(768).astype(np.float32))
        labels.append(f"label_{i % 10}")

    print(f"Total records: {len(ids)}")
    print(f"Embedding shape: {embeddings[0].shape}")

    # Batch operations
    batch_embeddings = embeddings[10:20]
    batch_labels = labels[10:20]

    print(f"\nBatch [10:20]:")
    print(f"  Embeddings: {len(batch_embeddings)} arrays")
    print(f"  Labels: {batch_labels[:3]}...")

    print("\n✓ ColumnVault example complete\n")


def example_4_vector_search():
    """Example 4: Vector similarity search."""
    print("=" * 80)
    print("Example 4: VectorKVault - Similarity Search")
    print("=" * 80)

    vkv = VectorKVault(":memory:", dimensions=128, metric="cosine")

    # Index documents
    print("Indexing 50 documents...")
    for i in range(50):
        embedding = np.random.randn(128).astype(np.float32)
        doc = f"Document {i}: Sample content about topic {i % 10}".encode()
        vkv.insert(embedding, doc)

    # Search
    query = np.random.randn(128).astype(np.float32)
    results = vkv.search(query, k=5)

    print(f"\nTop 5 results:")
    for rank, (id, distance, doc) in enumerate(results, 1):
        print(f"  {rank}. Distance={distance:.4f} | {doc.decode()[:50]}...")

    print("\n✓ Vector search complete\n")


def main():
    """Run all basic examples."""
    print("\n" + "=" * 80)
    print("KohakuVault Basic Usage")
    print("AUTO-PACKING ENABLED BY DEFAULT - Store any Python object!")
    print("=" * 80)
    print()

    example_1_kvault_auto_pack()
    example_2_kvault_cache()
    example_3_column_vault_vectors()
    example_4_vector_search()

    print("=" * 80)
    print("Basic Examples Complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  ✓ Auto-packing: Store numpy, dict, list, int, float, str automatically")
    print("  ✓ Auto-decoding: Get back Python objects, not bytes")
    print("  ✓ MessagePack for dicts/lists (efficient!)")
    print("  ✓ Media files stay raw (jpeg, mp4, etc.)")
    print("  ✓ Cache for performance")
    print("  ✓ Vector columns in ColumnVault")
    print("  ✓ Similarity search with VectorKVault")
    print("\nNo manual pickle/msgpack needed - it just works!")


if __name__ == "__main__":
    main()
