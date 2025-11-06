"""
CSBTree vs Python dict - Realistic Workload Benchmark

Tests realistic scenarios:
1. Timestamp-based keys (sorted by time)
2. Tuple keys (multi-dimensional sorting)
3. String keys (names, IDs)
4. Variable-length values (realistic data sizes)

Each operation tested separately:
- Insert
- Lookup
- Iteration
- Get sorted keys
- Range queries
"""

import argparse
import random
import string
import time
from datetime import datetime, timedelta

from kohakuvault import CSBTree, SkipList


def format_time(seconds: float) -> str:
    """Format time."""
    if seconds < 0.001:
        return f"{seconds*1000000:7.1f}us"
    elif seconds < 1.0:
        return f"{seconds*1000:7.2f}ms"
    else:
        return f"{seconds:7.2f}s"


def format_ops(ops_per_sec: float) -> str:
    """Format ops/sec."""
    if ops_per_sec > 1_000_000:
        return f"{ops_per_sec/1_000_000:6.2f}M/s"
    elif ops_per_sec > 1_000:
        return f"{ops_per_sec/1_000:6.2f}K/s"
    else:
        return f"{ops_per_sec:6.2f}/s"


class RealisticBenchmark:
    def __init__(self, n: int = 10000, order: int = 15):
        self.n = n
        self.order = order

    def run(self):
        print("=" * 100)
        print("CSBTree vs dict - Realistic Workload Benchmark")
        print("=" * 100)
        print(f"Dataset: {self.n:,} items | Tree order: {self.order}")
        print()

        self.scenario_timestamps()
        self.scenario_tuples()
        self.scenario_strings()
        self.scenario_variable_data()

    def scenario_timestamps(self):
        """Scenario 1: Timestamp-based logging/events"""
        print("\n" + "=" * 100)
        print("SCENARIO 1: Timestamp-based Event Log")
        print("Keys: datetime objects (timestamps)")
        print("Values: Variable-length strings (event messages)")
        print("=" * 100)

        # Generate random timestamps
        random.seed(42)
        base_time = datetime(2024, 1, 1)
        timestamps = [
            base_time + timedelta(seconds=random.randint(0, 1000000)) for _ in range(self.n)
        ]
        timestamps_shuffled = timestamps.copy()
        random.shuffle(timestamps_shuffled)

        # Generate variable-length messages
        messages = [f"Event_{i}_" + "x" * random.randint(10, 100) for i in range(self.n)]

        # Benchmark
        print()
        print(
            f"{'Operation':<30} | {'dict':>12} | {'CSBTree':>12} | {'SkipList':>12} | {'Ratio (Tree)':>15} | {'Ratio (Skip)':>15}"
        )
        print("-" * 110)

        # Insert
        test_dict = {}
        start = time.perf_counter()
        for ts, msg in zip(timestamps_shuffled, messages):
            test_dict[ts] = msg
        dict_insert = time.perf_counter() - start

        tree = CSBTree(order=self.order)
        start = time.perf_counter()
        for ts, msg in zip(timestamps_shuffled, messages):
            tree[ts] = msg
        tree_insert = time.perf_counter() - start

        skiplist = SkipList()
        start = time.perf_counter()
        for ts, msg in zip(timestamps_shuffled, messages):
            skiplist[ts] = msg
        skip_insert = time.perf_counter() - start

        tree_ratio = (
            f"{tree_insert/dict_insert:.2f}x slower"
            if tree_insert > dict_insert
            else f"{dict_insert/tree_insert:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_insert/dict_insert:.2f}x slower"
            if skip_insert > dict_insert
            else f"{dict_insert/skip_insert:.2f}x faster"
        )
        print(
            f"{'Insert (random order)':<30} | {format_time(dict_insert):>12} | {format_time(tree_insert):>12} | {format_time(skip_insert):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Get sorted (by timestamp)
        start = time.perf_counter()
        _ = sorted(test_dict.keys())
        dict_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = tree.keys()
        tree_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = skiplist.keys()
        skip_sorted = time.perf_counter() - start

        tree_ratio = (
            f"{dict_sorted/tree_sorted:.2f}x faster"
            if tree_sorted < dict_sorted
            else f"{tree_sorted/dict_sorted:.2f}x slower"
        )
        skip_ratio = (
            f"{dict_sorted/skip_sorted:.2f}x faster"
            if skip_sorted < dict_sorted
            else f"{skip_sorted/dict_sorted:.2f}x slower"
        )
        print(
            f"{'Get sorted timestamps':<30} | {format_time(dict_sorted):>12} | {format_time(tree_sorted):>12} | {format_time(skip_sorted):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Range query (events in time window)
        start_ts = timestamps[len(timestamps) // 4]
        end_ts = timestamps[len(timestamps) // 2]

        start = time.perf_counter()
        result = {k: v for k, v in test_dict.items() if start_ts <= k < end_ts}
        dict_range = time.perf_counter() - start

        start = time.perf_counter()
        result = tree.range(start_ts, end_ts)
        tree_range = time.perf_counter() - start

        start = time.perf_counter()
        result = skiplist.range(start_ts, end_ts)
        skip_range = time.perf_counter() - start

        tree_ratio = (
            f"{dict_range/tree_range:.2f}x faster"
            if tree_range < dict_range
            else f"{tree_range/dict_range:.2f}x slower"
        )
        skip_ratio = (
            f"{dict_range/skip_range:.2f}x faster"
            if skip_range < dict_range
            else f"{skip_range/dict_range:.2f}x slower"
        )
        print(
            f"{'Range query (time window)':<30} | {format_time(dict_range):>12} | {format_time(tree_range):>12} | {format_time(skip_range):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

    def scenario_tuples(self):
        """Scenario 2: Multi-dimensional keys (coordinates, IDs)"""
        print("\n" + "=" * 100)
        print("SCENARIO 2: Multi-dimensional Data (User ID, Timestamp)")
        print("Keys: (user_id, timestamp) tuples")
        print("Values: Variable-length dict objects")
        print("=" * 100)

        # Generate (user_id, timestamp) tuples
        random.seed(43)
        keys = [(random.randint(1, 1000), random.randint(0, 1000000)) for _ in range(self.n)]
        keys_shuffled = keys.copy()
        random.shuffle(keys_shuffled)

        # Generate variable data
        values = [{"data": "x" * random.randint(10, 50), "count": i} for i in range(self.n)]

        print()
        print(
            f"{'Operation':<30} | {'dict':>12} | {'CSBTree':>12} | {'SkipList':>12} | {'Ratio (Tree)':>15} | {'Ratio (Skip)':>15}"
        )
        print("-" * 110)

        # Insert
        test_dict = {}
        start = time.perf_counter()
        for k, v in zip(keys_shuffled, values):
            test_dict[k] = v
        dict_insert = time.perf_counter() - start

        tree = CSBTree(order=self.order)
        start = time.perf_counter()
        for k, v in zip(keys_shuffled, values):
            tree[k] = v
        tree_insert = time.perf_counter() - start

        skiplist = SkipList()
        start = time.perf_counter()
        for k, v in zip(keys_shuffled, values):
            skiplist[k] = v
        skip_insert = time.perf_counter() - start

        tree_ratio = (
            f"{tree_insert/dict_insert:.2f}x slower"
            if tree_insert > dict_insert
            else f"{dict_insert/tree_insert:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_insert/dict_insert:.2f}x slower"
            if skip_insert > dict_insert
            else f"{dict_insert/skip_insert:.2f}x faster"
        )
        print(
            f"{'Insert (random order)':<30} | {format_time(dict_insert):>12} | {format_time(tree_insert):>12} | {format_time(skip_insert):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Lookup
        n_lookup = min(1000, self.n)
        test_keys = random.sample(keys, n_lookup)

        start = time.perf_counter()
        for k in test_keys:
            _ = test_dict[k]
        dict_lookup = time.perf_counter() - start

        start = time.perf_counter()
        for k in test_keys:
            _ = tree[k]
        tree_lookup = time.perf_counter() - start

        start = time.perf_counter()
        for k in test_keys:
            _ = skiplist[k]
        skip_lookup = time.perf_counter() - start

        tree_ratio = (
            f"{tree_lookup/dict_lookup:.2f}x slower"
            if tree_lookup > dict_lookup
            else f"{dict_lookup/tree_lookup:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_lookup/dict_lookup:.2f}x slower"
            if skip_lookup > dict_lookup
            else f"{dict_lookup/skip_lookup:.2f}x faster"
        )
        print(
            f"{'Lookup (' + str(n_lookup) + ' keys)':<30} | {format_time(dict_lookup):>12} | {format_time(tree_lookup):>12} | {format_time(skip_lookup):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Get sorted (automatically sorts by user_id then timestamp)
        start = time.perf_counter()
        _ = sorted(test_dict.keys())
        dict_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = tree.keys()
        tree_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = skiplist.keys()
        skip_sorted = time.perf_counter() - start

        tree_ratio = (
            f"{dict_sorted/tree_sorted:.2f}x faster"
            if tree_sorted < dict_sorted
            else f"{tree_sorted/dict_sorted:.2f}x slower"
        )
        skip_ratio = (
            f"{dict_sorted/skip_sorted:.2f}x faster"
            if skip_sorted < dict_sorted
            else f"{skip_sorted/dict_sorted:.2f}x slower"
        )
        print(
            f"{'Get sorted tuples':<30} | {format_time(dict_sorted):>12} | {format_time(tree_sorted):>12} | {format_time(skip_sorted):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

    def scenario_strings(self):
        """Scenario 3: String keys (names, URLs, paths)"""
        print("\n" + "=" * 100)
        print("SCENARIO 3: String Keys (File Paths)")
        print("Keys: Strings (simulated file paths)")
        print("Values: Variable-length bytes (file metadata)")
        print("=" * 100)

        # Generate random file paths
        random.seed(44)

        def random_path():
            depth = random.randint(2, 5)
            parts = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(depth)
            ]
            return "/" + "/".join(parts)

        paths = [random_path() for _ in range(self.n)]
        paths_shuffled = paths.copy()
        random.shuffle(paths_shuffled)

        # Variable-length metadata
        metadata = [
            b"metadata_" + bytes([random.randint(0, 255) for _ in range(random.randint(20, 200))])
            for _ in range(self.n)
        ]

        print()
        print(
            f"{'Operation':<30} | {'dict':>12} | {'CSBTree':>12} | {'SkipList':>12} | {'Ratio (Tree)':>15} | {'Ratio (Skip)':>15}"
        )
        print("-" * 110)

        # Insert
        test_dict = {}
        start = time.perf_counter()
        for path, meta in zip(paths_shuffled, metadata):
            test_dict[path] = meta
        dict_insert = time.perf_counter() - start

        tree = CSBTree(order=self.order)
        start = time.perf_counter()
        for path, meta in zip(paths_shuffled, metadata):
            tree[path] = meta
        tree_insert = time.perf_counter() - start

        skiplist = SkipList()
        start = time.perf_counter()
        for path, meta in zip(paths_shuffled, metadata):
            skiplist[path] = meta
        skip_insert = time.perf_counter() - start

        tree_ratio = (
            f"{tree_insert/dict_insert:.2f}x slower"
            if tree_insert > dict_insert
            else f"{dict_insert/tree_insert:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_insert/dict_insert:.2f}x slower"
            if skip_insert > dict_insert
            else f"{dict_insert/skip_insert:.2f}x faster"
        )
        print(
            f"{'Insert (random order)':<30} | {format_time(dict_insert):>12} | {format_time(tree_insert):>12} | {format_time(skip_insert):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Get sorted paths
        start = time.perf_counter()
        _ = sorted(test_dict.keys())
        dict_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = tree.keys()
        tree_sorted = time.perf_counter() - start

        start = time.perf_counter()
        _ = skiplist.keys()
        skip_sorted = time.perf_counter() - start

        tree_ratio = (
            f"{dict_sorted/tree_sorted:.2f}x faster"
            if tree_sorted < dict_sorted
            else f"{tree_sorted/dict_sorted:.2f}x slower"
        )
        skip_ratio = (
            f"{dict_sorted/skip_sorted:.2f}x faster"
            if skip_sorted < dict_sorted
            else f"{skip_sorted/dict_sorted:.2f}x slower"
        )
        print(
            f"{'Get sorted paths':<30} | {format_time(dict_sorted):>12} | {format_time(tree_sorted):>12} | {format_time(skip_sorted):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Prefix search (all files under /abc/)
        prefix_paths = [p for p in paths if p.startswith("/a")]
        if len(prefix_paths) > 10:
            prefix_start = min(prefix_paths)
            prefix_end = prefix_start.rstrip("/") + "0"  # Hacky upper bound

            start = time.perf_counter()
            result = [v for k, v in test_dict.items() if k.startswith("/a")]
            dict_prefix = time.perf_counter() - start

            start = time.perf_counter()
            result = tree.range(prefix_start, prefix_end)
            tree_prefix = time.perf_counter() - start

            start = time.perf_counter()
            result = skiplist.range(prefix_start, prefix_end)
            skip_prefix = time.perf_counter() - start

            tree_ratio = (
                f"{dict_prefix/tree_prefix:.2f}x faster"
                if tree_prefix < dict_prefix
                else f"{tree_prefix/dict_prefix:.2f}x slower"
            )
            skip_ratio = (
                f"{dict_prefix/skip_prefix:.2f}x faster"
                if skip_prefix < dict_prefix
                else f"{skip_prefix/dict_prefix:.2f}x slower"
            )
            print(
                f"{'Prefix search (/a*)':<30} | {format_time(dict_prefix):>12} | {format_time(tree_prefix):>12} | {format_time(skip_prefix):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
            )

    def scenario_variable_data(self):
        """Scenario 4: Variable-length data"""
        print("\n" + "=" * 100)
        print("SCENARIO 4: Variable-Length Data")
        print("Keys: Integers (IDs)")
        print("Values: Variable-length lists/dicts (realistic objects)")
        print("=" * 100)

        # Generate data
        random.seed(45)
        keys = list(range(self.n))
        random.shuffle(keys)

        # Variable-length complex objects
        values = []
        for i in range(self.n):
            size = random.randint(1, 20)
            obj = {
                "id": i,
                "data": [random.random() for _ in range(size)],
                "tags": ["tag_" + str(random.randint(0, 100)) for _ in range(random.randint(1, 5))],
                "metadata": {"key": "value_" + str(i)},
            }
            values.append(obj)

        print()
        print(
            f"{'Operation':<30} | {'dict':>12} | {'CSBTree':>12} | {'SkipList':>12} | {'Ratio (Tree)':>15} | {'Ratio (Skip)':>15}"
        )
        print("-" * 110)

        # Insert
        test_dict = {}
        start = time.perf_counter()
        for k, v in zip(keys, values):
            test_dict[k] = v
        dict_insert = time.perf_counter() - start

        tree = CSBTree(order=self.order)
        start = time.perf_counter()
        for k, v in zip(keys, values):
            tree[k] = v
        tree_insert = time.perf_counter() - start

        skiplist = SkipList()
        start = time.perf_counter()
        for k, v in zip(keys, values):
            skiplist[k] = v
        skip_insert = time.perf_counter() - start

        tree_ratio = (
            f"{tree_insert/dict_insert:.2f}x slower"
            if tree_insert > dict_insert
            else f"{dict_insert/tree_insert:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_insert/dict_insert:.2f}x slower"
            if skip_insert > dict_insert
            else f"{dict_insert/skip_insert:.2f}x faster"
        )
        print(
            f"{'Insert (random order)':<30} | {format_time(dict_insert):>12} | {format_time(tree_insert):>12} | {format_time(skip_insert):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        # Iterate and access values
        start = time.perf_counter()
        total = sum(len(v["data"]) for v in test_dict.values())
        dict_iter = time.perf_counter() - start

        start = time.perf_counter()
        total = sum(len(v["data"]) for k, v in tree)
        tree_iter = time.perf_counter() - start

        start = time.perf_counter()
        total = sum(len(v["data"]) for k, v in skiplist)
        skip_iter = time.perf_counter() - start

        tree_ratio = (
            f"{tree_iter/dict_iter:.2f}x slower"
            if tree_iter > dict_iter
            else f"{dict_iter/tree_iter:.2f}x faster"
        )
        skip_ratio = (
            f"{skip_iter/dict_iter:.2f}x slower"
            if skip_iter > dict_iter
            else f"{dict_iter/skip_iter:.2f}x faster"
        )
        print(
            f"{'Iterate & process values':<30} | {format_time(dict_iter):>12} | {format_time(tree_iter):>12} | {format_time(skip_iter):>12} | {tree_ratio:>15} | {skip_ratio:>15}"
        )

        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSBTree realistic benchmark")
    parser.add_argument("--ops", type=int, default=10000, help="Number of items (default: 10000)")
    parser.add_argument("--order", type=int, default=15, help="Tree order (default: 15)")

    args = parser.parse_args()

    bench = RealisticBenchmark(n=args.ops, order=args.order)
    bench.run()

    print("\n✅ Benchmark complete!")
