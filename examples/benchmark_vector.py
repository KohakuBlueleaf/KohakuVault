"""Equivalent VectorKVault workload for 0.9.0 and current builds.

Run each package in a separate process with --package-root. Inputs are prepared
outside timers; inserts include commits; exact cosine search is unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
import tempfile
import threading
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.dimensions, args.queries, args.repeats) <= 0:
        parser.error("all workload sizes must be positive")
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))
    import numpy as np
    from kohakuvault import VectorKVault
    import kohakuvault._kvault as native

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(args.rows, args.dimensions)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_indices = np.arange(args.queries) % args.rows
    queries = vectors[query_indices].copy()
    values = [{"id": i} for i in range(args.rows)]
    samples = {"insert_committed": [], "get_by_id": [], "search": [], "four_readers": []}
    with tempfile.TemporaryDirectory() as directory:
        for repeat in range(args.repeats):
            vault = VectorKVault(
                str(Path(directory) / f"trial-{repeat}.db"), dimensions=args.dimensions
            )
            try:
                start = perf_counter()
                ids = [vault.insert(vector, value) for vector, value in zip(vectors, values)]
                samples["insert_committed"].append(perf_counter() - start)
                assert vault.count() == args.rows
                assert vault.get_by_id(ids[0])[1] == values[0]
                start = perf_counter()
                fetched = [vault.get_by_id(row_id) for row_id in ids]
                samples["get_by_id"].append(perf_counter() - start)
                for index, (vector, value) in enumerate(fetched):
                    np.testing.assert_allclose(vector, vectors[index], atol=1e-6)
                    assert value == values[index]
                vault.search(queries[0], k=10)
                start = perf_counter()
                found = [vault.search(query, k=10) for query in queries]
                samples["search"].append(perf_counter() - start)
                assert [result[0][2]["id"] for result in found] == query_indices.tolist()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    barrier = threading.Barrier(5)

                    def read():
                        barrier.wait(timeout=15)
                        return [vault.search(query, k=10)[0][2]["id"] for query in queries]

                    jobs = [pool.submit(read) for _ in range(4)]
                    start = perf_counter()
                    barrier.wait(timeout=15)
                    found = [job.result(timeout=120) for job in jobs]
                    samples["four_readers"].append(perf_counter() - start)
                    assert all(result == query_indices.tolist() for result in found)
            finally:
                if hasattr(vault, "close"):
                    vault.close()
                else:
                    # Old wrapper lacks close; release native ownership explicitly.
                    del vault._vault
                del vault
                gc.collect()
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "extension": str(Path(native.__file__).resolve()),
        "sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "workload": {
            "rows": args.rows,
            "dimensions": args.dimensions,
            "queries": args.queries,
            "k": 10,
            "repeats": args.repeats,
        },
        "seconds_median": {name: median(value) for name, value in samples.items()},
        "samples": samples,
        "notes": "Warm reads, fixed seeded float32 inputs, unchanged default WAL/NORMAL and exact cosine search. Four-reader workload has 4x queries. Input setup, validation and cleanup excluded. No explicit checkpoint timing included.",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
