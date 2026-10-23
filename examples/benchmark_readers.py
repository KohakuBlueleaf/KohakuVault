"""Measure KV reader retention, page-cache allocation and snapshot overhead.

    python examples/benchmark_readers.py --rows 10000 --repeats 3

Uses a fresh temporary database, warm reads and unchanged WAL/NORMAL defaults.
SQLite cache bytes exclude mmap, Python payloads, statements and active readers;
this is NOT process RSS. No hard active-reader limit is introduced.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median
import tempfile
import threading
from time import perf_counter

from kohakuvault import KVault


def run(rows, repeats):
    keys = [str(i).encode() for i in range(rows)]
    payload = b"x" * 256
    results = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readers.db"
        with KVault(path) as writer:
            writer.put_many([(key, payload) for key in keys])
            writer.checkpoint()
        for cache_kb, mmap_size, idle_limit in [
            (128, 0, 8),
            (8192, 0, 8),
            (8192, 268435456, 8),
            (8192, 268435456, 1),
            (8192, 268435456, 0),
        ]:
            samples = {"point": [], "one_key_snapshot_batch": [], "four_readers": []}
            with KVault(
                path,
                read_only=True,
                reader_cache_kb=cache_kb,
                mmap_size=mmap_size,
                reader_idle_limit=idle_limit,
            ) as vault:
                native = vault._native()
                for _ in range(repeats):
                    assert native.get_many(keys) == [payload] * rows
                    start = perf_counter()
                    values = [native.get(key) for key in keys]
                    samples["point"].append(perf_counter() - start)
                    assert values == [payload] * rows
                    start = perf_counter()
                    values = [native.get_many([key])[0] for key in keys]
                    samples["one_key_snapshot_batch"].append(perf_counter() - start)
                    assert values == [payload] * rows
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        barrier = threading.Barrier(5)

                        def read():
                            barrier.wait(timeout=10)
                            return [native.get(key) for key in keys]

                        jobs = [pool.submit(read) for _ in range(4)]
                        start = perf_counter()
                        barrier.wait(timeout=10)
                        values = [job.result(timeout=60) for job in jobs]
                        samples["four_readers"].append(perf_counter() - start)
                        assert all(value == [payload] * rows for value in values)
                results.append(
                    {
                        "reader_cache_kb": cache_kb,
                        "mmap_size": mmap_size,
                        "idle_limit": idle_limit,
                        "seconds_median": {name: median(value) for name, value in samples.items()},
                        "samples": samples,
                        "memory": vault.reader_pool_stats(),
                    }
                )
                del native  # Release the low-level reference before Windows cleanup.
    return {
        "rows": rows,
        "payload_bytes": 256,
        "repeats": repeats,
        "notes": [
            "Native point versus one-key bulk includes batch dispatch overhead as well as explicit snapshot cost; not a pure SQL microbenchmark.",
            "Memory sampled after all readers return. SQLite page-cache allocation only; not peak RSS or mmap residency.",
            "Four-reader time is 4x rows; no assertion of linear scaling. Input construction outside timers, validation after.",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.repeats) <= 0:
        parser.error("rows and repeats must be positive")
    result = run(args.rows, args.repeats)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
