"""Small CPU-only demo of arrays, logging, and independent readers."""

from pathlib import Path
import tempfile

import numpy as np

from kohakuvault import ColumnVault, KVault, VectorKVault


def main():
    with tempfile.TemporaryDirectory(prefix="kohakuvault-demo-") as directory:
        path = Path(directory) / "dataset.db"
        with KVault(path) as metadata, ColumnVault(path, block_bytes=64 << 10) as columns:
            metadata["dataset"] = {"name": "demo", "dimensions": 16}
            ids = columns.create_column("id", "i64")
            vectors = columns.create_column("embedding", "vec:f32:16")
            messages = columns.create_column("message", "str:utf8")

            # Write complete array batches without per-coordinate Python objects.
            ids.append_array(np.arange(1000, dtype=np.int64))
            embeddings = np.random.default_rng(42).standard_normal((1000, 16)).astype(np.float32)
            vectors.append_array(embeddings)
            print("Vector batch:", vectors.read_array(10, 5).shape)
            print("Selected IDs:", ids.take([0, 42, -1]).tolist())
            print(
                "Rows scanned:", sum(len(batch) for batch in vectors.iter_batches(batch_size=128))
            )

            # Buffer small logging calls and publish them together on exit.
            with messages.cache(cap_bytes=64 << 10):
                for index in range(100):
                    messages.append(f"processed batch {index}")
            payload, offsets = messages.read_buffers(0, 3)
            print("Packed messages:", len(payload), "bytes; offsets:", offsets.tolist())

            # An existing reader sees subsequent committed appends.
            with ColumnVault(path, read_only=True) as reader:
                observed = reader["message"]
                print("Before append:", len(observed))
                messages.append("finished")
                print("After append:", len(observed), observed[-1])

            # Similarity search retains the existing KV/vector file format.
            with VectorKVault(path, dimensions=16, metric="l2") as search:
                for index in range(10):
                    search.insert(embeddings[index], {"sample": index})
                print("Nearest sample:", search.search(embeddings[3], k=1)[0][2])

            print("Column blocks:", vectors.storage_info())
            print("KV metadata:", dict(metadata.items()))

    print("Demo complete; all handles closed and temporary files removed.")


if __name__ == "__main__":
    main()
