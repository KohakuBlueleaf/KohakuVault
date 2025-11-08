"""
Example: Vector Similarity Search with Numpy Arrays

This example demonstrates how to use VectorKVault for semantic search
using numpy arrays as the standard interface.
"""

import numpy as np
from kohakuvault import VectorKVault


def main():
    # Create VectorKVault for 128-dimensional embeddings
    print("Creating VectorKVault with 128 dimensions...")
    vkv = VectorKVault("vectors.db", dimensions=128, metric="cosine")

    # Generate some random embeddings (in practice, these would come from a model)
    print("\nInserting vectors...")
    embeddings = []
    for i in range(100):
        # Create random embedding
        vec = np.random.randn(128).astype(np.float32)

        # Store with metadata
        doc_id = vkv.insert(vec, f"Document {i}".encode())
        embeddings.append(vec)

        if (i + 1) % 20 == 0:
            print(f"  Inserted {i + 1} vectors")

    print(f"\nTotal vectors in database: {len(vkv)}")

    # Perform similarity search
    print("\n--- Similarity Search ---")
    query = np.random.randn(128).astype(np.float32)
    results = vkv.search(query, k=5)

    print(f"Top 5 similar vectors:")
    for rank, (id, distance, value) in enumerate(results, 1):
        print(f"  {rank}. ID={id}, Distance={distance:.4f}, Value={value.decode()}")

    # Get single most similar (KVault-like interface)
    print("\n--- Single Closest Match ---")
    closest = vkv.get(query)
    print(f"Closest match: {closest.decode()}")

    # Retrieve vector by ID (returns numpy array)
    print("\n--- Retrieve by ID ---")
    first_id = results[0][0]
    retrieved_vec, retrieved_val = vkv.get_by_id(first_id)

    print(f"ID {first_id}:")
    print(f"  Vector type: {type(retrieved_vec)}")
    print(f"  Vector shape: {retrieved_vec.shape}")
    print(f"  Vector dtype: {retrieved_vec.dtype}")
    print(f"  Value: {retrieved_val.decode()}")

    # Update a vector
    print("\n--- Update Vector ---")
    new_vec = np.random.randn(128).astype(np.float32)
    vkv.update(first_id, vector=new_vec)
    print(f"Updated vector for ID {first_id}")

    # Verify update
    updated_vec, _ = vkv.get_by_id(first_id)
    print(f"  New vector shape: {updated_vec.shape}")
    print(f"  Vectors are different: {not np.array_equal(retrieved_vec, updated_vec)}")

    # Demonstrate different metrics
    print("\n--- Different Metrics ---")

    # L2 (Euclidean distance)
    vkv_l2 = VectorKVault("vectors_l2.db", table="l2_vectors", dimensions=128, metric="l2")
    for i in range(10):
        vec = np.random.randn(128).astype(np.float32)
        vkv_l2.insert(vec, f"L2 Doc {i}".encode())

    query = np.random.randn(128).astype(np.float32)
    l2_results = vkv_l2.search(query, k=3)
    print(f"L2 metric - Top 3 results:")
    for rank, (id, distance, value) in enumerate(l2_results, 1):
        print(f"  {rank}. Distance={distance:.4f}, Value={value.decode()}")

    # Database info
    print("\n--- Database Info ---")
    info = vkv.info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
