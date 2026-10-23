"""Small column benchmark: committed writes, array reads, slices, and byte records.

Run after a release build: python examples/benchmark_columnar.py --rows 1000000
This is a warm-process smoke benchmark, not a cold-disk or durability benchmark.
"""

import argparse
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np

from kohakuvault import ColumnVault


def measure(results, name, payload_bytes, operation):
    start = perf_counter()
    value = operation()
    seconds = perf_counter() - start
    results[name] = {
        "seconds": seconds,
        "payload_bytes": payload_bytes,
        "mib_per_second": payload_bytes / (1 << 20) / seconds if seconds else 0,
    }
    print(f"{name:28s} {seconds * 1000:10.3f} ms  {results[name]['mib_per_second']:10.2f} MiB/s")
    return value


def benchmark(rows: int, block_bytes: int) -> dict:
    results = {}
    values = np.arange(rows, dtype=np.int64)
    list_values = values.tolist()

    with tempfile.TemporaryDirectory(prefix="kohakuvault-benchmark-") as directory:
        with ColumnVault(Path(directory) / "benchmark.db", block_bytes=block_bytes) as vault:
            numbers = vault.create_column("numbers", "i64")
            measure(results, "array_append", values.nbytes, lambda: numbers.append_array(values))
            actual = measure(results, "array_read", values.nbytes, numbers.read_array)
            np.testing.assert_array_equal(actual, values)

            copied = vault.create_column("from_list", "i64")
            measure(results, "list_extend", values.nbytes, lambda: copied.extend(list_values))
            actual = measure(results, "list_read", values.nbytes, lambda: copied[:])
            assert actual == list_values

            count = min(64, rows)
            requests = 1000

            def read_slices():
                for index in range(requests):
                    start = (index * 997) % (rows - count + 1)
                    actual = numbers[start : start + count]
                    assert actual[0] == start and actual[-1] == start + count - 1

            measure(results, "1000_small_slices", requests * count * 8, read_slices)

            cached = vault.create_column("cached", "i64")
            cached_rows = min(rows, 20_000)

            def append_cached():
                with cached.cache(cap_bytes=1 << 20):
                    for index in range(cached_rows):
                        cached.append(index)

            measure(results, "cached_scalar_append", cached_rows * 8, append_cached)
            assert len(cached) == cached_rows

            payloads = vault.create_column("payloads", "bytes")
            record_count = min(rows, 10_000)
            records = [b"x" * 64] * record_count
            measure(results, "variable_append", record_count * 64, lambda: payloads.extend(records))
            actual = measure(results, "variable_list_read", record_count * 64, lambda: payloads[:])
            assert actual == records
            payload, offsets = measure(
                results, "variable_buffer_read", record_count * 64, payloads.read_buffers
            )
            assert len(payload) == record_count * 64 and len(offsets) == record_count + 1

            # Checkpoint time is separate from already-committed append timings.
            measure(results, "checkpoint", 0, vault.checkpoint)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rows": rows,
        "block_bytes": block_bytes,
        "journal": "WAL",
        "synchronous": "NORMAL",
        "notes": "Input construction excluded; cache flush included; warm-process reads; one run.",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--block-bytes", type=int, default=1 << 20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.block_bytes <= 0:
        parser.error("rows and block-bytes must be positive")
    result = benchmark(args.rows, args.block_bytes)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
