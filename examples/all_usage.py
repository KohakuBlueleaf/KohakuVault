"""
KohakuVault Complete Feature Demonstration

This script demonstrates ALL features of KohakuVault including:
- Phase 1: Vector storage and similarity search
- Phase 2: Header system and metadata tracking
- Advanced features: Caching, streaming, slicing, bulk operations

Use this as a reference for building your own applications.
"""

import tempfile
from pathlib import Path

import numpy as np

from kohakuvault import ColumnVault, CSBTree, DataPacker, KVault, SkipList, VectorKVault


def demo_1_kvault_advanced():
    """Demonstration 1: Advanced KVault features."""
    print("\n" + "=" * 100)
    print("DEMO 1: KVault - Advanced Key-Value Storage")
    print("=" * 100)

    # Create persistent database
    db_path = "demo_kvault.db"
    kv = KVault(db_path)

    print("\n1.1 Basic Operations")
    print("-" * 50)

    # Store different types of binary data
    kv["config:json"] = b'{"server": "localhost", "port": 8080}'
    kv["image:001"] = b"\xff\xd8\xff\xe0" + b"fake jpeg data" * 100
    kv["video:001"] = b"fake mp4 data" * 1000

    print(f"Stored 3 items")
    print(f"Total size: {sum(len(kv[k]) for k in kv.keys())} bytes")

    print("\n1.2 Caching for Performance")
    print("-" * 50)

    # Enable cache for bulk writes
    kv.enable_cache(cap_bytes=32 * 1024 * 1024, flush_threshold=8 * 1024 * 1024)

    # Bulk write (cached in memory)
    print("Writing 10,000 key-value pairs with cache...")
    for i in range(10000):
        kv[f"bulk:item:{i:06d}"] = f"Value {i} with some data".encode() * 10

    # Flush to disk
    flushed = kv.flush_cache()
    print(f"Flushed {flushed:,} entries to disk")

    print(f"Total keys in vault: {len(kv):,}")

    print("\n1.3 Iteration and Filtering")
    print("-" * 50)

    # Get all keys with prefix
    bulk_keys = [k for k in kv.keys() if k.startswith(b"bulk:item:")]
    print(f"Keys with 'bulk:item:' prefix: {len(bulk_keys):,}")

    print("\n1.4 Header System (Phase 2)")
    print("-" * 50)

    # Enable header tracking
    kv.enable_headers()
    print(f"Headers enabled: {kv.headers_enabled()}")

    # Media files still stored as raw bytes (external tool compatible!)
    kv["photo.jpg"] = b"\xff\xd8\xff\xe0" + b"jpeg" * 50
    retrieved = kv["photo.jpg"]
    print(f"JPEG stored as raw (no header): {not retrieved.startswith(b'\\x89K')}")
    print(f"Can be opened by external tools: Yes!")

    kv.disable_headers()

    kv.close()
    print(f"\n✓ Database saved to: {db_path}")


def demo_2_column_vault_vectors():
    """Demonstration 2: ColumnVault with vector storage."""
    print("\n" + "=" * 100)
    print("DEMO 2: ColumnVault - Vector/Array Storage (Phase 1)")
    print("=" * 100)

    cv = ColumnVault("demo_vectors.db")

    print("\n2.1 Text Embeddings (768-dim float32)")
    print("-" * 50)

    # Create embedding column
    embeddings = cv.create_column("bert_embeddings", "vec:f32:768")

    # Add embeddings
    print("Adding 1,000 BERT embeddings...")
    vectors = [np.random.randn(768).astype(np.float32) for _ in range(1000)]
    embeddings.extend(vectors)  # Bulk operation

    print(f"Total embeddings: {len(embeddings):,}")
    # Each 768-dim f32 vector: 1 byte type + 768*4 bytes data = 3073 bytes
    bytes_per_embedding = 1 + 768 * 4
    print(f"Storage per embedding: {bytes_per_embedding:,} bytes")
    print(f"Total storage: ~{bytes_per_embedding * len(embeddings) / (1024*1024):.2f} MB")

    # Slice operations
    batch = embeddings[100:110]
    print(f"Batch [100:110]: {len(batch)} embeddings retrieved")

    print("\n2.2 Image Dataset (28x28 grayscale)")
    print("-" * 50)

    images = cv.create_column("mnist_digits", "vec:u8:28:28")

    # Add images
    print("Adding 1,000 MNIST-style images...")
    mnist_images = [np.random.randint(0, 256, (28, 28), dtype=np.uint8) for _ in range(1000)]
    images.extend(mnist_images)

    print(f"Total images: {len(images):,}")
    print(f"Image shape: {images[0].shape}")
    # Each 28x28 u8 image: 1 byte type + 28*28 bytes data = 785 bytes
    bytes_per_image = 1 + 28 * 28
    print(f"Storage per image: {bytes_per_image:,} bytes")

    print("\n2.3 Multi-Modal Data (combining types)")
    print("-" * 50)

    # Create mixed columns
    ids = cv.create_column("ids", "i64")
    labels = cv.create_column("labels", "str:utf8")
    features = cv.create_column("features", "vec:f32:128")
    metadata = cv.create_column("metadata", "msgpack")

    # Add data
    print("Adding 100 multi-modal records...")
    for i in range(100):
        ids.append(i)
        labels.append(f"class_{i % 10}")
        features.append(np.random.randn(128).astype(np.float32))
        metadata.append({"timestamp": i * 1000, "source": f"source_{i % 3}"})

    # Query
    print(f"\nRecord 50:")
    print(f"  ID: {ids[50]}")
    print(f"  Label: {labels[50]}")
    print(f"  Feature shape: {features[50].shape}")
    print(f"  Metadata: {metadata[50]}")

    print(f"\n✓ Database saved to: demo_vectors.db")


def demo_3_vector_similarity_search():
    """Demonstration 3: VectorKVault for semantic search."""
    print("\n" + "=" * 100)
    print("DEMO 3: VectorKVault - Vector Similarity Search")
    print("=" * 100)

    print("\n3.1 Document Search (Cosine Similarity)")
    print("-" * 50)

    # Create VectorKVault for document search
    doc_search = VectorKVault("demo_search.db", dimensions=384, metric="cosine")

    # Simulate adding documents with embeddings
    print("Indexing 500 documents...")
    documents = [
        f"Document {i}: This document contains information about topic {i % 20}."
        for i in range(500)
    ]

    for i, doc in enumerate(documents):
        # In real app, this would be from a model like BERT
        embedding = np.random.randn(384).astype(np.float32)
        doc_search.insert(embedding, doc.encode())

    print(f"Indexed {len(doc_search):,} documents")

    # Search
    query_embedding = np.random.randn(384).astype(np.float32)
    results = doc_search.search(query_embedding, k=10)

    print(f"\nTop 10 similar documents:")
    for rank, (id, distance, doc_bytes) in enumerate(results, 1):
        doc = doc_bytes.decode()
        print(f"  {rank:2d}. Distance={distance:.4f} | {doc[:60]}...")

    # KVault-like get (single closest)
    closest = doc_search.get(query_embedding)
    print(f"\nClosest document: {closest.decode()[:80]}...")

    print("\n3.2 Image Search (L2 Distance)")
    print("-" * 50)

    # Image feature search with L2 metric
    image_search = VectorKVault("demo_search.db", table="image_search", dimensions=512, metric="l2")

    # Add image features
    print("Indexing 200 image features...")
    for i in range(200):
        feature_vec = np.random.randn(512).astype(np.float32)
        image_id = f"image:{i:04d}".encode()
        image_search.insert(feature_vec, image_id)

    # Search
    query_feature = np.random.randn(512).astype(np.float32)
    image_results = image_search.search(query_feature, k=5)

    print(f"Top 5 similar images:")
    for rank, (id, distance, img_id) in enumerate(image_results, 1):
        print(f"  {rank}. Distance={distance:.4f} | {img_id.decode()}")

    # Get vector and metadata by ID
    vec, img_id = image_search.get_by_id(image_results[0][0])
    print(f"\nRetrieved vector shape: {vec.shape}, dtype: {vec.dtype}")

    # Update operations
    print("\n3.3 Update Operations")
    print("-" * 50)

    # Update vector
    new_embedding = np.random.randn(384).astype(np.float32)
    doc_search.update(results[0][0], vector=new_embedding)
    print(f"Updated vector for ID {results[0][0]}")

    # Update value
    doc_search.update(results[0][0], value=b"Updated document content")
    print(f"Updated value for ID {results[0][0]}")

    # Delete
    doc_search.delete(results[-1][0])
    print(f"Deleted ID {results[-1][0]}, remaining: {len(doc_search):,}")

    print(f"\n✓ Database saved to: demo_search.db")


def demo_4_datapacker_deep_dive():
    """Demonstration 4: DataPacker comprehensive features."""
    print("\n" + "=" * 100)
    print("DEMO 4: DataPacker - Deep Dive into Serialization")
    print("=" * 100)

    print("\n4.1 Primitive Types")
    print("-" * 50)

    # i64, f64
    for dtype in ["i64", "f64"]:
        packer = DataPacker(dtype)
        value = 12345 if dtype == "i64" else 123.45
        packed = packer.pack(value)
        print(f"{dtype}: {len(packed)} bytes")

    print("\n4.2 String Encodings")
    print("-" * 50)

    # Different encodings
    text = "Hello 世界 🚀"
    for encoding in ["utf8", "utf16le", "ascii"]:
        if encoding == "ascii":
            text_to_encode = "Hello World"
        else:
            text_to_encode = text

        packer = DataPacker(f"str:{encoding}")
        packed = packer.pack(text_to_encode)
        unpacked = packer.unpack(packed, 0)
        print(f"str:{encoding}: {len(packed)} bytes → '{unpacked}'")

    print("\n4.3 Fixed vs Variable Size")
    print("-" * 50)

    # Fixed size
    packer_fixed = DataPacker("str:32:utf8")
    packed_fixed = packer_fixed.pack("short")
    print(f"Fixed size (str:32): {len(packed_fixed)} bytes (padded)")

    # Variable size
    packer_var = DataPacker("str:utf8")
    packed_var = packer_var.pack("short")
    print(f"Variable size (str:utf8): {len(packed_var)} bytes (exact)")

    print("\n4.4 Vector Types (All Dimensions)")
    print("-" * 50)

    vector_types = [
        ("vec:f32:64", (64,), "Small embedding"),
        ("vec:f32:768", (768,), "BERT embedding"),
        ("vec:i64:10:20", (10, 20), "2D matrix"),
        ("vec:u8:3:224:224", (3, 224, 224), "RGB image"),
    ]

    for dtype, shape, description in vector_types:
        packer = DataPacker(dtype)

        if len(shape) == 1:
            data = np.random.randn(shape[0]).astype(
                np.float32 if "f32" in dtype else (np.uint8 if "u8" in dtype else np.int64)
            )
        else:
            data = (
                np.random.randint(0, 256, shape, dtype=np.uint8)
                if "u8" in dtype
                else np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
            )

        packed = packer.pack(data)
        data_size = np.prod(shape) * (4 if "f32" in dtype else (8 if "i64" in dtype else 1))
        overhead = len(packed) - data_size
        print(
            f"{description:20s} | {dtype:20s} | {len(packed):,} bytes | overhead: {overhead} bytes"
        )

    print("\n4.5 Structured Data (MessagePack vs JSON)")
    print("-" * 50)

    data = {
        "user_id": 12345,
        "name": "Alice Johnson",
        "scores": [95, 87, 92, 88, 91],
        "metadata": {"active": True, "premium": False},
    }

    # MessagePack (binary, compact)
    packer_msgpack = DataPacker("msgpack")
    packed_msgpack = packer_msgpack.pack(data)
    print(f"MessagePack: {len(packed_msgpack)} bytes")

    # For comparison with JSON
    import json

    json_bytes = json.dumps(data).encode()
    print(f"JSON: {len(json_bytes)} bytes")
    print(f"MessagePack saves: {100 * (1 - len(packed_msgpack)/len(json_bytes)):.1f}%")

    print("\n4.6 Bulk Operations Performance")
    print("-" * 50)

    # Bulk pack/unpack (optimized!)
    packer = DataPacker("vec:f32:256")
    vectors = [np.random.randn(256).astype(np.float32) for _ in range(1000)]

    print("Packing 1,000 vectors...")
    packed_all = packer.pack_many(vectors)
    print(f"  Total size: {len(packed_all):,} bytes")

    print("Unpacking 1,000 vectors...")
    unpacked_all = packer.unpack_many(packed_all, count=1000)
    print(f"  Unpacked: {len(unpacked_all):,} arrays")
    print(f"  First vector shape: {unpacked_all[0].shape}")

    print("\n✓ DataPacker deep dive complete")


def demo_5_combined_storage():
    """Demonstration 5: Combining KVault, ColumnVault, and VectorKVault."""
    print("\n" + "=" * 100)
    print("DEMO 5: Combined Storage - Real-World Scenario")
    print("=" * 100)
    print("Scenario: Building a semantic search system for documents")
    print()

    # All components share the same database file!
    db_file = "semantic_search_system.db"

    print("5.1 Document Storage (KVault)")
    print("-" * 50)

    # Store full document content in KVault
    kv = KVault(db_file, table="documents")
    kv.enable_cache()

    print("Storing 100 full documents...")
    for i in range(100):
        doc_id = f"doc:{i:04d}"
        content = f"Document {i}: " + f"content word " * 100
        kv[doc_id] = content.encode()

    kv.flush_cache()
    print(f"Stored {len(kv):,} documents")

    print("\n5.2 Metadata Storage (ColumnVault)")
    print("-" * 50)

    # Store metadata in columns
    cv = ColumnVault(db_file)

    doc_ids = cv.create_column("doc_ids", "i64")
    titles = cv.create_column("titles", "str:utf8")
    timestamps = cv.create_column("timestamps", "i64")
    categories = cv.create_column("categories", "msgpack")

    print("Adding metadata for 100 documents...")
    for i in range(100):
        doc_ids.append(i)
        titles.append(f"Document Title {i}")
        timestamps.append(1700000000 + i * 3600)
        categories.append({"primary": f"cat_{i % 5}", "tags": [f"tag_{i % 3}", f"tag_{i % 7}"]})

    print(f"Metadata columns: {cv.list_columns()}")

    print("\n5.3 Vector Search Index (VectorKVault)")
    print("-" * 50)

    # Build search index
    vkv = VectorKVault(db_file, table="search_index", dimensions=384, metric="cosine")

    print("Building search index with 100 embeddings...")
    for i in range(100):
        # Simulate embedding from sentence transformer
        embedding = np.random.randn(384).astype(np.float32)
        doc_ref = f"doc:{i:04d}".encode()
        vkv.insert(embedding, doc_ref)

    print(f"Search index: {len(vkv):,} vectors")

    print("\n5.4 End-to-End Search")
    print("-" * 50)

    # Query
    query_text = "find documents about machine learning"
    # Simulate query embedding
    query_embedding = np.random.randn(384).astype(np.float32)

    # Search similar documents
    results = vkv.search(query_embedding, k=5)

    print(f"Query: '{query_text}'")
    print(f"\nTop 5 results:")

    for rank, (vec_id, distance, doc_ref_bytes) in enumerate(results, 1):
        doc_ref = doc_ref_bytes.decode()
        doc_idx = int(doc_ref.split(":")[1])

        # Get full document from KVault
        full_doc = kv[doc_ref].decode()

        # Get metadata from ColumnVault
        title = titles[doc_idx]
        category = categories[doc_idx]

        print(f"\n  {rank}. Distance={distance:.4f}")
        print(f"     Title: {title}")
        print(f"     Category: {category['primary']}")
        print(f"     Content: {full_doc[:80]}...")

    print(f"\n✓ Semantic search system complete")
    print(f"✓ All data in single database: {db_file}")
    print(f"   - {len(kv):,} documents (KVault)")
    print(f"   - {len(doc_ids):,} metadata records (ColumnVault)")
    print(f"   - {len(vkv):,} vectors (VectorKVault)")


def demo_6_ordered_containers():
    """Demonstration 6: Ordered containers (CSBTree, SkipList)."""
    print("\n" + "=" * 100)
    print("DEMO 6: Ordered Containers - CSBTree and SkipList")
    print("=" * 100)

    print("\n6.1 CSBTree (Cache-Sensitive B+Tree)")
    print("-" * 50)

    tree = CSBTree(order=15)

    # Insert in random order
    import random

    keys = list(range(100))
    random.shuffle(keys)

    for key in keys:
        tree[key] = f"value_{key}"

    print(f"Inserted {len(tree)} items in random order")
    print(f"Tree automatically maintains sorted order!")

    # Iterate in sorted order
    sorted_keys = tree.keys()
    print(f"First 10 sorted keys: {sorted_keys[:10]}")

    # Range queries
    range_results = tree.range(25, 35)
    range_keys = [k for k, v in range_results]
    print(f"Range [25, 35): {range_keys}")

    print("\n6.2 SkipList (Lock-Free Concurrent)")
    print("-" * 50)

    skiplist = SkipList()

    # Insert
    for i in range(50):
        skiplist[i * 2] = f"even_{i}"

    print(f"Inserted {len(skiplist)} items")
    print(f"First 5 keys: {list(skiplist.keys())[:5]}")

    # Range query
    range_items = skiplist.range(10, 30)
    range_keys = [k for k, v in range_items]
    print(f"Range [10, 30): {range_keys}")

    print("\n✓ Ordered containers complete")


def demo_7_streaming_large_files():
    """Demonstration 7: Streaming large files with KVault."""
    print("\n" + "=" * 100)
    print("DEMO 7: Streaming Large Files")
    print("=" * 100)

    kv = KVault("demo_streaming.db")

    # Create a large fake file
    print("\n7.1 Writing Large File via Streaming")
    print("-" * 50)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        # Write 10MB file
        tmp.write(b"x" * (10 * 1024 * 1024))

    try:
        print(f"Streaming 10MB file to KVault...")
        with open(tmp_path, "rb") as f:
            kv.put_file("large_file", f)

        file_size = len(kv["large_file"])
        print(f"Stored file size: {file_size / (1024*1024):.2f} MB")

        print("\n7.2 Reading Large File via Streaming")
        print("-" * 50)

        output_path = "retrieved_large_file.bin"
        with open(output_path, "wb") as f:
            kv.get_to_file("large_file", f)

        retrieved_size = Path(output_path).stat().st_size
        print(f"Retrieved file size: {retrieved_size / (1024*1024):.2f} MB")
        print(f"Sizes match: {file_size == retrieved_size}")

        # Cleanup
        Path(output_path).unlink()

    finally:
        Path(tmp_path).unlink()

    kv.close()
    print(f"\n✓ Streaming example complete")


def demo_8_performance_best_practices():
    """Demonstration 8: Performance optimization tips."""
    print("\n" + "=" * 100)
    print("DEMO 8: Performance Best Practices")
    print("=" * 100)

    print("\n8.1 KVault Caching")
    print("-" * 50)
    print("✓ Use enable_cache() for bulk writes (10-100x faster)")
    print("✓ Flush after batch operations")
    print("✓ Use context manager: with kv.cache(): ...")

    print("\n8.2 ColumnVault Bulk Operations")
    print("-" * 50)
    print("✓ Use extend() instead of multiple append() calls")
    print("✓ Use slice operations col[10:20] instead of loops")
    print("✓ Pack_many/unpack_many for DataPacker: 3-35x faster")

    print("\n8.3 Vector Storage")
    print("-" * 50)
    print("✓ Use fixed-shape vectors (vec:f32:768) for minimal overhead")
    print("✓ Arbitrary shape (vec:f32) for flexibility, 6-byte overhead")
    print("✓ Bulk operations: up to 37x faster for unpack_many")

    print("\n8.4 VectorKVault Tips")
    print("-" * 50)
    print("✓ Choose metric wisely: cosine for text, L2 for images")
    print("✓ Batch inserts for better SQLite performance")
    print("✓ Consider dimensions: 128-768 for best balance")

    print("\n✓ Best practices summary complete")


def cleanup_demo_files():
    """Clean up demo database files."""
    for db_file in [
        "demo_kvault.db",
        "demo_vectors.db",
        "demo_search.db",
        "demo_streaming.db",
        "ml_project.db",
        "semantic_search_system.db",
    ]:
        for suffix in ["", "-shm", "-wal"]:
            path = Path(db_file + suffix)
            if path.exists():
                path.unlink()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 100)
    print("KohakuVault Complete Feature Demonstration")
    print("Showcasing all capabilities from Phase 1 & Phase 2")
    print("=" * 100)

    try:
        demo_1_kvault_advanced()
        demo_2_column_vault_vectors()
        demo_3_vector_similarity_search()
        demo_4_datapacker_deep_dive()
        demo_6_ordered_containers()
        demo_7_streaming_large_files()
        demo_8_performance_best_practices()

        print("\n" + "=" * 100)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 100)
        print("\nWhat You've Learned:")
        print("  ✓ KVault: Key-value storage with caching and streaming")
        print("  ✓ ColumnVault: Type-safe columnar storage")
        print("  ✓ DataPacker: Efficient serialization (primitives, vectors, structured)")
        print("  ✓ VectorKVault: Fast vector similarity search")
        print("  ✓ Header System: Encoding detection (ready for auto-packing)")
        print("  ✓ Ordered Containers: CSBTree and SkipList")
        print("  ✓ Performance: Caching, bulk operations, streaming")
        print("\nReady to build your application with KohakuVault!")

    finally:
        print("\n" + "=" * 100)
        print("Cleaning up demo files...")
        cleanup_demo_files()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
