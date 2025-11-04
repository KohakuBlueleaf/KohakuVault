"""
KohakuVault Performance Benchmark

Simplified benchmark suite with configurable parameters.
Tests: KVault write/read, ColumnVault append/extend/read, DataPacker
"""

import os
import time
import tempfile
import shutil
from typing import List
from tqdm import tqdm
from kohakuvault import KVault, ColumnVault, DataPacker


# =============================================================================
# Configuration
# =============================================================================


class BenchmarkConfig:
    """Configurable benchmark parameters."""

    def __init__(
        self,
        entries: List[int] = None,
        entry_sizes: List[int] = None,
        cache_sizes_mb: List[int] = None,
        column_sizes: List[int] = None,  # Column-specific data sizes (for bytes columns)
        min_chunk_kb: int = 16,  # Minimum chunk size in KB (default 16KB)
        max_chunk_mb: int = 16,  # Maximum chunk size in MB (default 16MB)
    ):
        # Use sorted(set()) to remove duplicates and sort
        self.entries = sorted(set(entries or [1000, 10000]))
        self.entry_sizes = sorted(set(entry_sizes or [1024, 16384]))  # 1KB, 16KB (for KVault)
        self.cache_sizes_mb = sorted(set(cache_sizes_mb or [0, 64]))  # No cache, 64MB cache
        self.column_sizes = sorted(set(column_sizes or []))  # Additional column sizes (bytes:N)
        self.min_chunk_kb = min_chunk_kb  # Min chunk size (default 16KB)
        self.max_chunk_mb = max_chunk_mb  # Max chunk size (default 16MB)
        self.temp_dir = tempfile.mkdtemp()

    def get_db_path(self, name: str) -> str:
        """Get path for disk database."""
        return os.path.join(self.temp_dir, f"{name}.db")

    def cleanup(self):
        """Remove all temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def format_time(seconds: float) -> str:
    """Format time in appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}us"
    elif seconds < 1.0:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.0f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


# =============================================================================
# 1. KVault Write Benchmarks
# =============================================================================


def benchmark_kvault_write(config: BenchmarkConfig):
    """Benchmark KVault write with/without cache."""
    print("\n" + "=" * 80)
    print("1. KVault Write Performance (with/without cache)")
    print("=" * 80)

    results = []

    # Generate test configurations
    test_configs = []
    for n_entries in config.entries:
        for entry_size in config.entry_sizes:
            for cache_mb in config.cache_sizes_mb:
                test_configs.append((n_entries, entry_size, cache_mb))

    # Run benchmarks with progress bar
    for n_entries, entry_size, cache_mb in tqdm(test_configs, desc="KVault Write"):
        db_path = config.get_db_path(f"kv_write_{n_entries}_{entry_size}_{cache_mb}")

        vault = KVault(db_path)

        if cache_mb > 0:
            vault.enable_cache(
                cap_bytes=cache_mb * 1024 * 1024, flush_threshold=cache_mb * 1024 * 1024 // 4
            )

        value = b"x" * entry_size

        # Benchmark write
        start = time.perf_counter()
        for i in range(n_entries):
            vault[f"k:{i:08d}"] = value

        if cache_mb > 0:
            vault.flush_cache()

        # Force WAL checkpoint to get accurate DB size
        vault.checkpoint()

        elapsed = time.perf_counter() - start
        vault.close()

        # Get DB size
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

        results.append(
            {
                "entries": n_entries,
                "entry_size": entry_size,
                "cache_mb": cache_mb,
                "time": elapsed,
                "ops_per_sec": n_entries / elapsed,
                "mb_per_sec": (n_entries * entry_size) / (1024 * 1024) / elapsed,
                "db_size": db_size,
            }
        )

    # Print results table
    print(
        f"\n{'Entries':<10s} {'Size':<10s} {'Cache':<10s} {'Time':<12s} {'Ops/Sec':<15s} {'MB/s':<10s} {'DB Size':<12s}"
    )
    print("-" * 80)
    for r in results:
        cache_str = f"{r['cache_mb']}MB" if r["cache_mb"] > 0 else "Disabled"
        print(
            f"{r['entries']:<10,d} {format_size(r['entry_size']):<10s} {cache_str:<10s} "
            f"{format_time(r['time']):<12s} {r['ops_per_sec']:<15,.0f} {r['mb_per_sec']:<10.1f} "
            f"{format_size(r['db_size']):<12s}"
        )


# =============================================================================
# 2. KVault Read Benchmarks
# =============================================================================


def benchmark_kvault_read(config: BenchmarkConfig):
    """Benchmark KVault read performance."""
    print("\n" + "=" * 80)
    print("2. KVault Read Performance")
    print("=" * 80)

    results = []

    # Generate test configurations
    test_configs = []
    for n_entries in config.entries:
        for entry_size in config.entry_sizes:
            test_configs.append((n_entries, entry_size))

    # Run benchmarks with progress bar
    for n_entries, entry_size in tqdm(test_configs, desc="KVault Read"):
        db_path = config.get_db_path(f"kv_read_{n_entries}_{entry_size}")

        # Setup: Write data first
        vault = KVault(db_path)
        vault.enable_cache()
        value = b"x" * entry_size
        for i in range(n_entries):
            vault[f"k:{i:08d}"] = value
        vault.flush_cache()
        vault.close()

        # Reopen and benchmark reads
        vault = KVault(db_path)

        start = time.perf_counter()
        for i in range(n_entries):
            _ = vault[f"k:{i:08d}"]
        elapsed = time.perf_counter() - start

        vault.close()

        results.append(
            {
                "entries": n_entries,
                "entry_size": entry_size,
                "time": elapsed,
                "ops_per_sec": n_entries / elapsed,
                "mb_per_sec": (n_entries * entry_size) / (1024 * 1024) / elapsed,
            }
        )

    # Print results table
    print(f"\n{'Entries':<10s} {'Size':<10s} {'Time':<12s} {'Ops/Sec':<15s} {'MB/s':<10s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['entries']:<10,d} {format_size(r['entry_size']):<10s} "
            f"{format_time(r['time']):<12s} {r['ops_per_sec']:<15,.0f} {r['mb_per_sec']:<10.1f}"
        )


# =============================================================================
# 3. ColumnVault Append/Extend Benchmarks
# =============================================================================


def benchmark_column_append_extend(config: BenchmarkConfig):
    """Benchmark ColumnVault append vs extend."""
    print("\n" + "=" * 80)
    print("3. ColumnVault Append vs Extend Performance")
    print("=" * 80)

    results = []

    # Always test i64, f64, small bytes, small string, msgpack, and user-specified sizes
    dtypes_to_test = [
        "i64",
        "f64",
        "bytes:32",  # Small fixed bytes (~32 bytes)
        "str:32:utf8",  # Small fixed string (32 bytes)
        "str:utf8",  # Variable string
        "msgpack",  # Structured data
    ]
    for size in config.column_sizes:
        dtypes_to_test.append(f"bytes:{size}")

    # Test configurations
    test_configs = []
    for n_entries in config.entries:
        for dtype in dtypes_to_test:
            # Use same entry count for both append and extend for fair comparison
            test_configs.append((n_entries, dtype, "append"))
            test_configs.append((n_entries, dtype, "extend"))

    # Run benchmarks with progress bar
    for n_entries, dtype, method in tqdm(test_configs, desc="Column Append/Extend"):
        # Sanitize dtype for filename and column name (replace : with _)
        safe_dtype = dtype.replace(":", "_")
        db_path = config.get_db_path(f"col_{method}_{n_entries}_{safe_dtype}")

        cv = ColumnVault(
            db_path,
            min_chunk_bytes=config.min_chunk_kb * 1024,
            max_chunk_bytes=config.max_chunk_mb * 1024 * 1024,
        )
        col = cv.create_column("test", dtype)

        # Prepare data and calculate ACTUAL packed size using DataPacker
        test_packer = DataPacker(dtype)

        if dtype == "i64":
            data = list(range(n_entries))
        elif dtype == "f64":
            data = [float(i) * 1.5 for i in range(n_entries)]
        elif dtype.startswith("bytes:"):
            size = int(dtype.split(":")[1])
            data = [b"x" * size for _ in range(n_entries)]
        elif dtype.startswith("str:") and ":" in dtype[4:]:  # str:N:encoding
            # Fixed-size string
            data = [f"str_{i:04d}" for i in range(n_entries)]
        elif dtype.startswith("str:"):
            # Variable-size string
            data = [f"string_{i}" for i in range(n_entries)]
        elif dtype == "msgpack":
            # Structured data
            data = [{"id": i, "val": i * 1.5} for i in range(n_entries)]
        else:
            data = list(range(n_entries))

        # Calculate actual packed size using DataPacker
        sample_packed = test_packer.pack(data[0])
        elem_size = len(sample_packed)  # ACTUAL encoded size

        # Benchmark
        start = time.perf_counter()
        if method == "append":
            for item in data:
                col.append(item)
        else:  # extend
            col.extend(data)
        elapsed = time.perf_counter() - start

        # Checkpoint WAL to get accurate DB size
        cv.checkpoint()

        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

        # Calculate throughput using actual encoded data size
        total_bytes = n_entries * elem_size  # Use actual packed size from DataPacker
        mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        results.append(
            {
                "entries": n_entries,
                "dtype": dtype,
                "method": method,
                "time": elapsed,
                "ops_per_sec": n_entries / elapsed,
                "mb_per_sec": mb_per_sec,
                "db_size": db_size,
            }
        )

    # Print results table with actual encoded data size
    print(
        f"\n{'Entries':<10s} {'Type':<18s} {'Method':<10s} {'Time':<12s} {'Ops/Sec':<15s} {'MB/s':<10s} {'Encoded/Entry':<14s} {'DB Size':<12s}"
    )
    print("-" * 100)

    # Group by entry count and dtype to show append/extend pairs
    for n_entries in config.entries:
        for dtype in dtypes_to_test:
            append_r = next(
                (
                    r
                    for r in results
                    if r["dtype"] == dtype and r["method"] == "append" and r["entries"] == n_entries
                ),
                None,
            )
            extend_r = next(
                (
                    r
                    for r in results
                    if r["dtype"] == dtype and r["method"] == "extend" and r["entries"] == n_entries
                ),
                None,
            )

            if append_r and extend_r:
                # Get actual encoded size from the result (calculated via DataPacker)
                # This is stored implicitly - recalculate from mb_per_sec
                encoded_bytes = (
                    append_r["mb_per_sec"] * (1024 * 1024) * append_r["time"]
                ) / append_r["entries"]

                print(
                    f"{append_r['entries']:<10,d} {dtype:<18s} {append_r['method']:<10s} "
                    f"{format_time(append_r['time']):<12s} {append_r['ops_per_sec']:<15,.0f} "
                    f"{append_r['mb_per_sec']:<10.1f} {encoded_bytes:<14.1f} {format_size(append_r['db_size']):<12s}"
                )

                speedup = extend_r["ops_per_sec"] / append_r["ops_per_sec"]
                encoded_bytes_extend = (
                    extend_r["mb_per_sec"] * (1024 * 1024) * extend_r["time"]
                ) / extend_r["entries"]
                print(
                    f"{extend_r['entries']:<10,d} {dtype:<18s} {extend_r['method']:<10s} "
                    f"{format_time(extend_r['time']):<12s} {extend_r['ops_per_sec']:<15,.0f} "
                    f"{extend_r['mb_per_sec']:<10.1f} {encoded_bytes_extend:<14.1f} {format_size(extend_r['db_size']):<12s} ({speedup:.1f}x)"
                )
                print()


# =============================================================================
# 4. ColumnVault Read Benchmarks
# =============================================================================


def benchmark_column_read(config: BenchmarkConfig):
    """Benchmark ColumnVault read and read_range."""
    print("\n" + "=" * 80)
    print("4. ColumnVault Read Performance (single & range)")
    print("=" * 80)

    results = []

    # Test i64, f64, bytes, strings, msgpack
    dtypes_to_test = [
        ("i64", 8),
        ("f64", 8),
        ("bytes:32", 32),
        ("str:32:utf8", 32),
        ("str:utf8", 10),  # Variable, ~10 bytes average
        ("msgpack", 32),  # Variable, ~32 bytes average
    ]

    # Test configurations
    test_configs = []
    for n_entries in config.entries:
        for dtype, elem_size in dtypes_to_test:
            test_configs.append((n_entries, dtype, elem_size, "single"))
            test_configs.append((n_entries, dtype, elem_size, "range"))

    # Run benchmarks with progress bar
    for n_entries, dtype, elem_size, read_type in tqdm(test_configs, desc="Column Read"):
        # Sanitize dtype for filename
        safe_dtype = dtype.replace(":", "_")
        db_path = config.get_db_path(f"col_read_{n_entries}_{safe_dtype}_{read_type}")

        # Setup: Create and populate column with configured chunk sizes
        cv = ColumnVault(
            db_path,
            min_chunk_bytes=config.min_chunk_kb * 1024,
            max_chunk_bytes=config.max_chunk_mb * 1024 * 1024,
        )
        col = cv.create_column("test", dtype)

        # Populate with appropriate data
        if dtype == "i64":
            col.extend(list(range(n_entries)))
        elif dtype == "f64":
            col.extend([float(i) * 1.5 for i in range(n_entries)])
        elif dtype == "bytes:32":
            col.extend([b"x" * 32 for _ in range(n_entries)])
        elif dtype == "str:32:utf8":
            col.extend([f"str_{i:04d}" for _ in range(n_entries)])
        elif dtype == "str:utf8":
            col.extend([f"string_{i}" for i in range(n_entries)])
        elif dtype == "msgpack":
            col.extend([{"id": i, "val": i * 1.5} for i in range(n_entries)])

        # Benchmark reads
        n_reads = min(n_entries, 1000)  # Read subset for large datasets

        start = time.perf_counter()
        if read_type == "single":
            # Random single reads
            import random

            random.seed(42)
            for _ in range(n_reads):
                idx = random.randint(0, n_entries - 1)
                _ = col[idx]
        else:  # range
            # Sequential range reads
            chunk_size = 100
            for start_idx in range(0, n_reads, chunk_size):
                end_idx = min(start_idx + chunk_size, n_entries)
                for i in range(start_idx, end_idx):
                    _ = col[i]

        elapsed = time.perf_counter() - start

        # Calculate throughput
        total_bytes = n_reads * elem_size
        mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        kb_per_sec = (total_bytes / 1024) / elapsed if elapsed > 0 else 0

        results.append(
            {
                "entries": n_entries,
                "dtype": dtype,
                "elem_size": elem_size,
                "read_type": read_type,
                "n_reads": n_reads,
                "time": elapsed,
                "ops_per_sec": n_reads / elapsed,
                "mb_per_sec": mb_per_sec,
                "kb_per_sec": kb_per_sec,
            }
        )

    # Print results table
    print(
        f"\n{'Entries':<10s} {'Type':<18s} {'Read Type':<12s} {'Reads':<10s} {'Time':<12s} {'Reads/Sec':<15s} {'KB/s':<10s}"
    )
    print("-" * 90)
    for r in results:
        # Add size info to dtype display
        dtype_display = r["dtype"]
        if r["dtype"] == "msgpack":
            dtype_display = "msgpack(~32B)"
        elif r["dtype"] == "str:utf8":
            dtype_display = "str:utf8(~10B)"
        elif r["dtype"].startswith("str:"):
            size = r["dtype"].split(":")[1]
            dtype_display = f"str:{size}B"

        # Use KB/s for small data, MB/s for large
        throughput = r["kb_per_sec"] if r["elem_size"] <= 32 else r["mb_per_sec"] * 1024
        print(
            f"{r['entries']:<10,d} {dtype_display:<18s} {r['read_type']:<12s} {r['n_reads']:<10,d} "
            f"{format_time(r['time']):<12s} {r['ops_per_sec']:<15,.0f} {throughput:<10.1f}"
        )


# =============================================================================
# 5. DataPacker Benchmarks
# =============================================================================


def benchmark_datapacker(config: BenchmarkConfig):
    """Benchmark DataPacker pack/unpack performance."""
    print("\n" + "=" * 80)
    print("5. DataPacker Performance (pack/unpack/pack_many/unpack_many)")
    print("=" * 80)

    results = []

    # Test different data types
    packer_types = [
        ("i64", lambda i: i),
        ("f64", lambda i: float(i)),
        ("bytes:128", lambda i: b"x" * 128),
        ("str:utf8", lambda i: f"string_{i}"),
        ("msgpack", lambda i: {"id": i, "name": f"item_{i}", "value": i * 1.5}),
    ]

    n_ops = 10000  # Fixed for packer tests

    for dtype, data_gen in tqdm(packer_types, desc="DataPacker"):
        packer = DataPacker(dtype)

        # Test pack
        data = [data_gen(i) for i in range(n_ops)]

        start = time.perf_counter()
        for item in data:
            _ = packer.pack(item)
        pack_elapsed = time.perf_counter() - start

        # Test pack_many (works for ALL types now)
        start = time.perf_counter()
        packed_all = packer.pack_many(data)
        pack_many_elapsed = time.perf_counter() - start

        # Test unpack
        packed_data = [packer.pack(item) for item in data]

        start = time.perf_counter()
        for packed in packed_data:
            _ = packer.unpack(packed, 0)
        unpack_elapsed = time.perf_counter() - start

        # Test unpack_many
        unpack_many_elapsed = None
        if packer.is_varsize:
            # For variable-size, calculate offsets
            offsets = []
            pos = 0
            for packed in packed_data:
                offsets.append(pos)
                pos += len(packed)

            start = time.perf_counter()
            _ = packer.unpack_many(packed_all, offsets=offsets)
            unpack_many_elapsed = time.perf_counter() - start
        else:
            # For fixed-size, use count
            start = time.perf_counter()
            _ = packer.unpack_many(packed_all, count=n_ops)
            unpack_many_elapsed = time.perf_counter() - start

        results.append(
            {
                "dtype": dtype,
                "pack_time": pack_elapsed,
                "pack_ops_per_sec": n_ops / pack_elapsed,
                "pack_many_time": pack_many_elapsed,
                "pack_many_ops_per_sec": n_ops / pack_many_elapsed,
                "unpack_time": unpack_elapsed,
                "unpack_ops_per_sec": n_ops / unpack_elapsed,
                "unpack_many_time": unpack_many_elapsed,
                "unpack_many_ops_per_sec": (
                    n_ops / unpack_many_elapsed if unpack_many_elapsed else None
                ),
            }
        )

    # Print timing results
    print(
        f"\n{'Type':<20s} {'Encoded Size':<15s} {'pack()':<12s} {'pack_many':<12s} {'unpack()':<12s} {'unpack_many':<12s}"
    )
    print("-" * 90)
    for r in results:
        # Get actual encoded size for first element
        dtype = r["dtype"]
        packer = DataPacker(dtype)

        # Generate sample data to measure size
        if dtype == "i64":
            sample = 0
        elif dtype == "f64":
            sample = 0.0
        elif dtype == "bytes:128":
            sample = b"x" * 128
        elif dtype == "str:utf8":
            sample = "string_0"
        elif dtype == "msgpack":
            sample = {"id": 0, "name": "item_0", "value": 0.0}

        sample_packed = packer.pack(sample)
        encoded_size = len(sample_packed)

        print(
            f"{dtype:<20s} "
            f"{encoded_size:<15d} "
            f"{format_time(r['pack_time']):<12s} "
            f"{format_time(r['pack_many_time']):<12s} "
            f"{format_time(r['unpack_time']):<12s} "
            f"{format_time(r['unpack_many_time']):<12s}"
        )

    print(
        f"\n{'Type':<20s} {'pack/s':<15s} {'pack_many/s':<15s} {'unpack/s':<15s} {'unpack_many/s':<15s}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['dtype']:<20s} "
            f"{r['pack_ops_per_sec']:<15,.0f} "
            f"{r['pack_many_ops_per_sec']:<15,.0f} "
            f"{r['unpack_ops_per_sec']:<15,.0f} "
            f"{r['unpack_many_ops_per_sec']:<15,.0f}"
        )


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def run_benchmark(
    entries: List[int] = None,
    entry_sizes: List[int] = None,
    cache_sizes_mb: List[int] = None,
    column_sizes: List[int] = None,
    min_chunk_kb: int = 16,
    max_chunk_mb: int = 16,
):
    """
    Run comprehensive benchmark suite.

    Parameters
    ----------
    entries : List[int]
        Number of entries to test, e.g., [1000, 10000]
    entry_sizes : List[int]
        Entry sizes in bytes for KVault, e.g., [1024, 16384]
    cache_sizes_mb : List[int]
        Cache sizes in MB, e.g., [0, 64]
    column_sizes : List[int]
        Additional column data sizes (bytes:N), e.g., [128, 1024]
    min_chunk_kb : int
        Minimum chunk size in KB (default: 16)
    max_chunk_mb : int
        Maximum chunk size in MB (default: 16)
    """
    config = BenchmarkConfig(
        entries, entry_sizes, cache_sizes_mb, column_sizes, min_chunk_kb, max_chunk_mb
    )

    print("=" * 80)
    print("KohakuVault Performance Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Entries: {config.entries}")
    print(f"  Entry sizes (KVault): {[format_size(s) for s in config.entry_sizes]}")
    print(f"  Cache sizes: {config.cache_sizes_mb} MB")
    print(f"  Column chunk: {config.min_chunk_kb}KB - {config.max_chunk_mb}MB")
    print(
        f"  Column sizes: i64, f64"
        + (
            f", bytes:[{', '.join(str(s) for s in config.column_sizes)}]"
            if config.column_sizes
            else ""
        )
    )
    print(f"  Temp dir: {config.temp_dir}")

    try:
        # Run all benchmark suites
        benchmark_kvault_write(config)
        benchmark_kvault_read(config)
        benchmark_column_append_extend(config)
        benchmark_column_read(config)
        benchmark_datapacker(config)

        print("\n" + "=" * 80)
        print("Benchmark Complete!")
        print("=" * 80)

    finally:
        config.cleanup()
        print(f"\nCleaned up temp directory: {config.temp_dir}")


def run_quick_benchmark():
    """Quick benchmark with minimal configurations."""
    print("Running quick benchmark (use custom parameters for full test)...\n")
    run_benchmark(
        entries=[1000],
        entry_sizes=[1024],
        cache_sizes_mb=[0, 64],
        column_sizes=[],  # Just i64/f64
        min_chunk_kb=16,  # 16KB
        max_chunk_mb=16,  # 16MB
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KohakuVault Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                                    # Quick benchmark
  python benchmark.py --entries 1000 10000              # Custom entry counts
  python benchmark.py --sizes 1024 16384 65536          # KVault entry sizes
  python benchmark.py --cache 0 16 64                    # Custom cache sizes
  python benchmark.py --column-sizes 128 1024 16384     # Test bytes:128, bytes:1024, bytes:16384
  python benchmark.py --min-chunk 128 --max-chunk 64    # 128KB min, 64MB max chunks
  python benchmark.py --entries 10000 --column-sizes 4096  # Specific config
        """,
    )

    parser.add_argument(
        "--entries",
        type=int,
        nargs="+",
        default=[1000, 10000],
        help="Number of entries to test (default: 1000 10000)",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 16384],
        help="Entry sizes in bytes (default: 1024 16384)",
    )

    parser.add_argument(
        "--cache",
        type=int,
        nargs="+",
        default=[0, 64],
        help="Cache sizes in MB (default: 0 64)",
    )

    parser.add_argument(
        "--column-sizes",
        type=int,
        nargs="*",
        default=[],
        help="Additional column data sizes for bytes:N testing (default: just i64/f64)",
    )

    parser.add_argument(
        "--min-chunk",
        type=int,
        default=16,
        help="Minimum chunk size in KB (default: 16)",
    )

    parser.add_argument(
        "--max-chunk",
        type=int,
        default=16,
        help="Maximum chunk size in MB (default: 16)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with minimal config",
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark()
    else:
        run_benchmark(
            entries=args.entries,
            entry_sizes=args.sizes,
            cache_sizes_mb=args.cache,
            column_sizes=args.column_sizes,
            min_chunk_kb=args.min_chunk,
            max_chunk_mb=args.max_chunk,
        )
