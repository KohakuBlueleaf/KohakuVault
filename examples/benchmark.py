"""
KohakuVault Performance Benchmark Matrix

Tests performance across multiple dimensions:
- Storage: In-memory vs Disk
- Operations: Write, Read, Random Access, Delete
- Data count: 100, 1K, 10K, 100K
- Value size: 100B, 1KB, 10KB, 100KB
- Cache: Disabled, 16MB, 64MB (KVault)
- Chunk size: Small, Medium, Large (ColumnVault)
"""

import os
import time
import tempfile
import shutil
from typing import Dict, Any
from kohakuvault import KVault, ColumnVault


# =============================================================================
# Benchmark Infrastructure
# =============================================================================


class BenchmarkRunner:
    """Manages benchmark execution and cleanup."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = []

    def get_db_path(self, name: str) -> str:
        """Get path for disk-based database."""
        return os.path.join(self.temp_dir, f"{name}.db")

    def cleanup(self):
        """Remove all temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def run(self, func, **params) -> Dict[str, Any]:
        """Time and execute a benchmark function."""
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start

        return {"elapsed": elapsed, "params": params, **result}


def format_time(seconds: float) -> str:
    """Format time in appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}µs"
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
# KVault Benchmarks
# =============================================================================


def bench_kvault_write(
    storage: str, n_ops: int, value_size: int, cache_mb: int, runner: BenchmarkRunner
):
    """Benchmark KVault write operations."""
    db_path = (
        ":memory:"
        if storage == "memory"
        else runner.get_db_path(f"kv_write_{n_ops}_{value_size}_{cache_mb}")
    )

    vault = KVault(db_path)

    if cache_mb > 0:
        vault.enable_cache(
            cap_bytes=cache_mb * 1024 * 1024, flush_threshold=cache_mb * 1024 * 1024 // 4
        )

    value = b"x" * value_size

    start = time.perf_counter()
    for i in range(n_ops):
        vault[f"k:{i:08d}"] = value

    if cache_mb > 0:
        vault.flush_cache()

    elapsed = time.perf_counter() - start

    # Get DB size for disk storage
    db_size = 0
    if storage == "disk":
        try:
            db_size = os.path.getsize(db_path)
        except:
            pass

    vault.close()

    return {
        "ops_per_sec": n_ops / elapsed if elapsed > 0 else 0,
        "mb_per_sec": (n_ops * value_size) / (1024 * 1024) / elapsed if elapsed > 0 else 0,
        "db_size": db_size,
    }


def bench_kvault_read(storage: str, n_ops: int, value_size: int, runner: BenchmarkRunner):
    """Benchmark KVault read operations."""
    # For memory, use shared memory so it persists across connections
    # For disk, use regular file path
    if storage == "memory":
        db_path = "file::memory:?cache=shared"  # Shared memory database
    else:
        db_path = runner.get_db_path(f"kv_read_{n_ops}_{value_size}")

    # Setup - write data
    vault = KVault(db_path)
    vault.enable_cache()
    value = b"x" * value_size
    for i in range(n_ops):
        vault[f"k:{i:08d}"] = value
    vault.flush_cache()

    # For memory storage, keep vault open to preserve data
    # For disk storage, can close and reopen
    if storage == "disk":
        vault.close()
        vault = KVault(db_path)

    # Benchmark reads
    start = time.perf_counter()

    for i in range(n_ops):
        _ = vault[f"k:{i:08d}"]

    elapsed = time.perf_counter() - start
    vault.close()

    return {
        "ops_per_sec": n_ops / elapsed if elapsed > 0 else 0,
        "mb_per_sec": (n_ops * value_size) / (1024 * 1024) / elapsed if elapsed > 0 else 0,
    }


# =============================================================================
# ColumnVault Benchmarks
# =============================================================================


def bench_column_append(
    storage: str, n_ops: int, dtype: str, min_kb: int, max_mb: int, runner: BenchmarkRunner
):
    """Benchmark ColumnVault individual append operations."""
    # Use unique path even for memory to avoid conflicts
    import random

    unique_id = random.randint(100000, 999999)
    # Use temp files for all benchmarks (memory vs disk just affects analysis)
    db_path = runner.get_db_path(
        f"col_append_{storage}_{n_ops}_{dtype}_{min_kb}_{max_mb}_{unique_id}"
    )

    cv = ColumnVault(db_path, min_chunk_bytes=min_kb * 1024, max_chunk_bytes=max_mb * 1024 * 1024)
    cv.create_column("test", dtype)
    col = cv["test"]

    start = time.perf_counter()

    if dtype == "i64":
        for i in range(n_ops):
            col.append(i)
    elif dtype == "f64":
        for i in range(n_ops):
            col.append(i * 1.5)
    elif dtype.startswith("bytes:"):
        size = int(dtype.split(":")[1])
        for i in range(n_ops):
            col.append(b"x" * size)
    elif dtype == "bytes":
        for i in range(n_ops):
            col.append(f"variable_string_{i}".encode())

    elapsed = time.perf_counter() - start

    db_size = 0
    if storage == "disk":
        try:
            db_size = os.path.getsize(db_path)
        except:
            pass

    return {"ops_per_sec": n_ops / elapsed if elapsed > 0 else 0, "db_size": db_size}


def bench_column_extend(
    storage: str, n_ops: int, dtype: str, min_kb: int, max_mb: int, runner: BenchmarkRunner
):
    """Benchmark ColumnVault bulk extend operations."""
    import random

    unique_id = random.randint(100000, 999999)
    db_path = runner.get_db_path(f"col_extend_{storage}_{n_ops}_{dtype}_{unique_id}")

    cv = ColumnVault(db_path, min_chunk_bytes=min_kb * 1024, max_chunk_bytes=max_mb * 1024 * 1024)
    cv.create_column("test", dtype)
    col = cv["test"]

    # Prepare data
    if dtype == "i64":
        data = list(range(n_ops))
    elif dtype == "f64":
        data = [i * 1.5 for i in range(n_ops)]
    elif dtype.startswith("bytes:"):
        size = int(dtype.split(":")[1])
        data = [b"x" * size for _ in range(n_ops)]
    elif dtype == "bytes":
        data = [f"variable_string_{i}".encode() for i in range(n_ops)]

    start = time.perf_counter()
    col.extend(data)
    elapsed = time.perf_counter() - start

    return {"ops_per_sec": n_ops / elapsed if elapsed > 0 else 0}


def bench_column_random_read(
    storage: str,
    n_elements: int,
    n_reads: int,
    dtype: str,
    min_kb: int,
    max_mb: int,
    runner: BenchmarkRunner,
):
    """Benchmark ColumnVault random access reads."""
    import random

    unique_id = random.randint(100000, 999999)
    db_path = runner.get_db_path(f"col_randread_{storage}_{n_elements}_{unique_id}")

    # Setup
    cv = ColumnVault(db_path, min_chunk_bytes=min_kb * 1024, max_chunk_bytes=max_mb * 1024 * 1024)
    cv.create_column("test", dtype)
    col = cv["test"]

    if dtype == "i64":
        col.extend(list(range(n_elements)))
    elif dtype == "f64":
        col.extend([i * 1.5 for i in range(n_elements)])

    # Random reads
    import random

    random.seed(42)
    indices = [random.randint(0, n_elements - 1) for _ in range(n_reads)]

    start = time.perf_counter()
    for idx in indices:
        _ = col[idx]
    elapsed = time.perf_counter() - start

    return {"ops_per_sec": n_reads / elapsed if elapsed > 0 else 0}


# =============================================================================
# Matrix Benchmark Runner
# =============================================================================


def get_scale_params(scale: str):
    """Get parameters based on scale level."""
    if scale == "small":
        return {
            "kv_ops": [1000],
            "kv_large_ops": [],
            "col_ops": [1000],
            "col_large_ops": [],
            "scale_test": [100, 1000],
        }
    elif scale == "large":
        return {
            "kv_ops": [1000, 10000],
            "kv_large_ops": [100000],
            "col_ops": [1000, 10000],
            "col_large_ops": [100000],
            "scale_test": [100, 1000, 10000, 100000],
        }
    else:  # medium (default)
        return {
            "kv_ops": [1000, 10000],
            "kv_large_ops": [],
            "col_ops": [1000, 10000],
            "col_large_ops": [],
            "scale_test": [100, 1000, 10000],
        }


def run_matrix_benchmark():
    """Run comprehensive matrix benchmark with default settings."""
    run_matrix_benchmark_configurable({1, 2, 3, 4, 5, 6, 7}, "medium")


def run_matrix_benchmark_configurable(matrices_to_run, scale="medium"):
    """Run configurable matrix benchmark."""
    runner = BenchmarkRunner()
    params = get_scale_params(scale)

    print("=" * 100)
    print("KohakuVault Performance Benchmark Matrix")
    print("=" * 100)
    print()

    # =============================================================================
    # Matrix 1: KVault Write - Storage × Data Count × Value Size × Cache
    # =============================================================================

    if 1 in matrices_to_run:
        print("MATRIX 1: KVault Write Performance")
    print("-" * 100)
    print()

    # Comprehensive matrix for different scenarios
    configs = [
        # (storage, n_ops, value_size, cache_mb)
        ("memory", 1000, 100, 0),
        ("memory", 1000, 100, 64),
        ("memory", 1000, 1024, 0),
        ("memory", 1000, 1024, 64),
        ("memory", 10000, 100, 0),
        ("memory", 10000, 100, 64),
        ("memory", 10000, 10240, 0),  # 10KB values
        ("memory", 10000, 10240, 64),
        ("disk", 1000, 100, 0),
        ("disk", 1000, 100, 64),
        ("disk", 1000, 1024, 0),
        ("disk", 1000, 1024, 64),
        ("disk", 1000, 131072, 0),  # 128KB values (image-like)
        ("disk", 10000, 1024, 0),
        ("disk", 10000, 1024, 0),
        ("disk", 100000, 1024, 0),  # 100K ops
    ]

    kv_write_results = []
    for storage, n_ops, value_size, cache_mb in configs:
        result = runner.run(
            lambda s=storage, n=n_ops, v=value_size, c=cache_mb: bench_kvault_write(
                s, n, v, c, runner
            ),
            storage=storage,
            n_ops=n_ops,
            value_size=value_size,
            cache_mb=cache_mb,
        )
        kv_write_results.append(result)

    # Print table
    print(
        f"{'Storage':<8s} {'Ops':<8s} {'ValSize':<8s} {'Cache':<8s} {'Time':<12s} {'Ops/Sec':<12s} {'MB/s':<10s} {'DB Size':<10s}"
    )
    print("-" * 100)
    for r in kv_write_results:
        p = r["params"]
        cache_str = f"{p['cache_mb']}MB" if p["cache_mb"] > 0 else "No"
        db_size_str = format_size(r["db_size"]) if r["db_size"] > 0 else "-"
        print(
            f"{p['storage']:<8s} {p['n_ops']:<8,d} {format_size(p['value_size']):<8s} {cache_str:<8s} "
            f"{format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} {r['mb_per_sec']:<10.1f} {db_size_str:<10s}"
        )

    print()

    # =============================================================================
    # Matrix 2: KVault Read - Storage × Data Count × Value Size
    # =============================================================================

    print("MATRIX 2: KVault Read Performance")
    print("-" * 100)
    print()

    read_configs = [
        ("memory", 1000, 100),
        ("memory", 1000, 1024),
        ("memory", 10000, 100),
        ("memory", 10000, 1024),
        ("memory", 10000, 10240),  # 10KB values
        ("disk", 1000, 100),
        ("disk", 1000, 1024),
        ("disk", 10000, 1024),
        ("disk", 10000, 10240),
        ("disk", 100000, 1024),  # 100K ops
    ]

    kv_read_results = []
    for storage, n_ops, value_size in read_configs:
        result = runner.run(
            lambda s=storage, n=n_ops, v=value_size: bench_kvault_read(s, n, v, runner),
            storage=storage,
            n_ops=n_ops,
            value_size=value_size,
        )
        kv_read_results.append(result)

    print(
        f"{'Storage':<8s} {'Ops':<8s} {'ValSize':<8s} {'Time':<12s} {'Ops/Sec':<12s} {'MB/s':<10s}"
    )
    print("-" * 100)
    for r in kv_read_results:
        p = r["params"]
        print(
            f"{p['storage']:<8s} {p['n_ops']:<8,d} {format_size(p['value_size']):<8s} "
            f"{format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} {r['mb_per_sec']:<10.1f}"
        )

    print()

    # =============================================================================
    # Matrix 3: ColumnVault Append - Storage × Data Count × Dtype × Chunk Size
    # =============================================================================

    print("MATRIX 3: ColumnVault Append Performance")
    print("-" * 100)
    print()

    col_append_configs = [
        # (storage, n_ops, dtype, min_kb, max_mb)
        ("memory", 1000, "i64", 16, 1),  # Tiny chunks
        ("memory", 1000, "i64", 128, 16),  # Default chunks
        ("memory", 1000, "i64", 512, 64),  # Large chunks
        ("memory", 10000, "i64", 16, 1),
        ("memory", 10000, "i64", 128, 16),
        ("memory", 10000, "i64", 512, 64),
        ("memory", 100000, "i64", 128, 16),  # 100K scale test
        ("disk", 1000, "i64", 128, 16),
        ("disk", 10000, "i64", 128, 16),
        ("disk", 10000, "f64", 128, 16),
        ("disk", 10000, "bytes:32", 128, 16),
        ("disk", 1000, "bytes:131072", 128, 16),  # Large data (image-like)
        ("disk", 1000, "bytes", 128, 16),  # Variable size
        ("disk", 100000, "i64", 128, 16),  # 100K scale test on disk
    ]

    col_append_results = []
    for storage, n_ops, dtype, min_kb, max_mb in col_append_configs:
        result = runner.run(
            lambda s=storage, n=n_ops, d=dtype, mi=min_kb, ma=max_mb: bench_column_append(
                s, n, d, mi, ma, runner
            ),
            storage=storage,
            n_ops=n_ops,
            dtype=dtype,
            min_kb=min_kb,
            max_mb=max_mb,
        )
        col_append_results.append(result)

    print(
        f"{'Storage':<8s} {'Ops':<8s} {'Type':<12s} {'Chunks':<15s} {'Time':<12s} {'Ops/Sec':<12s} {'DB Size':<10s}"
    )
    print("-" * 100)
    for r in col_append_results:
        p = r["params"]
        chunk_str = f"{p['min_kb']}KB-{p['max_mb']}MB"
        db_size_str = format_size(r["db_size"]) if r["db_size"] > 0 else "-"
        print(
            f"{p['storage']:<8s} {p['n_ops']:<8,d} {p['dtype']:<12s} {chunk_str:<15s} "
            f"{format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} {db_size_str:<10s}"
        )

    print()

    # =============================================================================
    # Matrix 4: ColumnVault Extend vs Append
    # =============================================================================

    print("MATRIX 4: ColumnVault Extend (bulk) vs Append (individual)")
    print("-" * 100)
    print()

    extend_configs = [
        ("memory", 1000, "i64", 128, 16),
        ("memory", 10000, "i64", 128, 16),
        ("memory", 100000, "i64", 128, 16),  # 100K ops
        ("disk", 1000, "i64", 128, 16),
        ("disk", 10000, "i64", 128, 16),
        ("disk", 100000, "i64", 128, 16),  # 100K ops
    ]

    print(
        f"{'Storage':<8s} {'Ops':<8s} {'Method':<10s} {'Time':<12s} {'Ops/Sec':<12s} {'Speedup':<10s}"
    )
    print("-" * 100)

    for storage, n_ops, dtype, min_kb, max_mb in extend_configs:
        # Append
        append_result = runner.run(
            lambda s=storage, n=n_ops, d=dtype, mi=min_kb, ma=max_mb: bench_column_append(
                s, n, d, mi, ma, runner
            ),
            storage=storage,
            n_ops=n_ops,
            dtype=dtype,
        )

        # Extend
        extend_result = runner.run(
            lambda s=storage, n=n_ops, d=dtype, mi=min_kb, ma=max_mb: bench_column_extend(
                s, n, d, mi, ma, runner
            ),
            storage=storage,
            n_ops=n_ops,
            dtype=dtype,
        )

        speedup = (
            extend_result["ops_per_sec"] / append_result["ops_per_sec"]
            if append_result["ops_per_sec"] > 0
            else 0
        )

        print(
            f"{storage:<8s} {n_ops:<8,d} {'append':<10s} {format_time(append_result['elapsed']):<12s} {append_result['ops_per_sec']:<12,.0f} {'-':<10s}"
        )
        print(
            f"{storage:<8s} {n_ops:<8,d} {'extend':<10s} {format_time(extend_result['elapsed']):<12s} {extend_result['ops_per_sec']:<12,.0f} {speedup:<10.1f}x"
        )
        print()

    # =============================================================================
    # Matrix 5: ColumnVault Random Read - Chunk Size Impact
    # =============================================================================

    print("MATRIX 5: ColumnVault Random Read Performance")
    print("-" * 100)
    print()

    rand_configs = [
        ("memory", 10000, 1000, "i64", 16, 1),
        ("memory", 10000, 1000, "i64", 128, 16),
        ("memory", 10000, 1000, "i64", 512, 64),
        ("memory", 100000, 5000, "i64", 128, 16),  # Larger scale
        ("disk", 10000, 1000, "i64", 16, 1),
        ("disk", 10000, 1000, "i64", 128, 16),
        ("disk", 10000, 1000, "i64", 512, 64),
        ("disk", 100000, 5000, "i64", 128, 16),  # Larger scale
    ]

    col_read_results = []
    for storage, n_elements, n_reads, dtype, min_kb, max_mb in rand_configs:
        result = runner.run(
            lambda s=storage, ne=n_elements, nr=n_reads, d=dtype, mi=min_kb, ma=max_mb: bench_column_random_read(
                s, ne, nr, d, mi, ma, runner
            ),
            storage=storage,
            n_elements=n_elements,
            n_reads=n_reads,
            chunk_config=f"{min_kb}KB-{max_mb}MB",
        )
        col_read_results.append(result)

    print(
        f"{'Storage':<8s} {'Elements':<10s} {'Reads':<8s} {'Chunks':<15s} {'Time':<12s} {'Reads/Sec':<12s} {'µs/Read':<10s}"
    )
    print("-" * 100)
    for r in col_read_results:
        p = r["params"]
        time_per_read = (r["elapsed"] / p["n_reads"] * 1000000) if p["n_reads"] > 0 else 0
        print(
            f"{p['storage']:<8s} {p['n_elements']:<10,d} {p['n_reads']:<8,d} {p['chunk_config']:<15s} "
            f"{format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} {time_per_read:<10.1f}"
        )

    print()

    # =============================================================================
    # Matrix 6: Data Type Comparison (same conditions)
    # =============================================================================

    print("MATRIX 6: Data Type Performance Comparison (disk, default chunks)")
    print("-" * 100)
    print()

    # Adjust operation count based on data size to keep runtime reasonable
    dtype_configs = [
        ("i64", 10000),
        ("f64", 10000),
        ("bytes:32", 10000),
        ("bytes:100", 10000),
        ("bytes:1024", 10000),
        ("bytes:10240", 1000),  # 10KB - document-like (fewer ops)
        ("bytes:131072", 100),  # 128KB - image-like (much fewer ops)
        ("bytes", 10000),  # Variable
    ]

    dtype_results = []
    for dtype, n_ops in dtype_configs:
        result = runner.run(
            lambda d=dtype, n=n_ops: bench_column_append("disk", n, d, 128, 16, runner),
            dtype=dtype,
            n_ops=n_ops,
        )
        dtype_results.append(result)

    print(
        f"{'Type':<15s} {'Ops':<8s} {'Time':<12s} {'Ops/Sec':<12s} {'DB Size':<10s} {'Bytes/Elem':<12s} {'MB/s':<10s}"
    )
    print("-" * 100)
    for i, r in enumerate(dtype_results):
        dtype, n_ops = dtype_configs[i]
        elem_size = (
            8 if dtype in ["i64", "f64"] else (int(dtype.split(":")[1]) if ":" in dtype else 20)
        )
        elem_size_str = str(elem_size) if isinstance(elem_size, int) else "~20"
        mb_per_sec = (n_ops * elem_size) / (1024 * 1024) / r["elapsed"] if r["elapsed"] > 0 else 0
        print(
            f"{dtype:<15s} {n_ops:<8,d} {format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} "
            f"{format_size(r['db_size']):<10s} {elem_size_str:<12s} {mb_per_sec:<10.1f}"
        )

    print()

    # =============================================================================
    # Matrix 7: Scale Test - How performance changes with data volume
    # =============================================================================

    print("MATRIX 7: Scale Test - ColumnVault i64 append (disk, 128KB-16MB chunks)")
    print("-" * 100)
    print()

    scale_configs = [100, 1000, 10000, 100000]

    scale_results = []
    for n in scale_configs:
        result = runner.run(
            lambda nn=n: bench_column_append("disk", nn, "i64", 128, 16, runner), n_ops=n
        )
        scale_results.append(result)

    print(f"{'N Elements':<12s} {'Time':<12s} {'Ops/Sec':<12s} {'Time/Op':<12s} {'DB Size':<10s}")
    print("-" * 100)
    for i, r in enumerate(scale_results):
        n = scale_configs[i]
        time_per_op = r["elapsed"] / n if n > 0 else 0
        print(
            f"{n:<12,d} {format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f} "
            f"{format_time(time_per_op):<12s} {format_size(r['db_size']):<10s}"
        )

    print()

    # =============================================================================
    # Summary Analysis
    # =============================================================================

    print("=" * 100)
    print("SUMMARY ANALYSIS")
    print("=" * 100)
    print()

    # Cache impact
    mem_nocache = [
        r
        for r in kv_write_results
        if r["params"]["storage"] == "memory"
        and r["params"]["cache_mb"] == 0
        and r["params"]["n_ops"] == 1000
        and r["params"]["value_size"] == 1024
    ][0]
    mem_cache = [
        r
        for r in kv_write_results
        if r["params"]["storage"] == "memory"
        and r["params"]["cache_mb"] == 64
        and r["params"]["n_ops"] == 1000
        and r["params"]["value_size"] == 1024
    ][0]
    cache_speedup = mem_cache["ops_per_sec"] / mem_nocache["ops_per_sec"]

    print(f"1. Cache Impact (KVault):")
    print(f"   - Without cache: {mem_nocache['ops_per_sec']:,.0f} ops/s")
    print(f"   - With 64MB cache: {mem_cache['ops_per_sec']:,.0f} ops/s")
    print(f"   - Speedup: {cache_speedup:.1f}x")
    print()

    # Storage impact
    disk_result = [
        r
        for r in kv_write_results
        if r["params"]["storage"] == "disk"
        and r["params"]["n_ops"] == 1000
        and r["params"]["value_size"] == 1024
        and r["params"]["cache_mb"] == 64
    ][0]
    storage_ratio = mem_cache["ops_per_sec"] / disk_result["ops_per_sec"]
    print(f"2. Storage Impact (KVault with cache):")
    print(f"   - Memory: {mem_cache['ops_per_sec']:,.0f} ops/s")
    print(f"   - Disk: {disk_result['ops_per_sec']:,.0f} ops/s")
    print(f"   - Memory is {storage_ratio:.1f}x faster")
    print()

    # Value size impact
    small_val = [
        r
        for r in kv_write_results
        if r["params"]["storage"] == "memory"
        and r["params"]["n_ops"] == 1000
        and r["params"]["value_size"] == 100
        and r["params"]["cache_mb"] == 0
    ][0]
    large_val = [
        r
        for r in kv_write_results
        if r["params"]["storage"] == "memory"
        and r["params"]["n_ops"] == 1000
        and r["params"]["value_size"] == 1024
        and r["params"]["cache_mb"] == 0
    ][0]
    print(f"3. Value Size Impact (KVault, memory, no cache):")
    print(
        f"   - 100B values: {small_val['ops_per_sec']:,.0f} ops/s ({small_val['mb_per_sec']:.1f} MB/s)"
    )
    print(
        f"   - 1KB values: {large_val['ops_per_sec']:,.0f} ops/s ({large_val['mb_per_sec']:.1f} MB/s)"
    )
    print(f"   - MB/s ratio: {large_val['mb_per_sec'] / small_val['mb_per_sec']:.1f}x")
    print()

    # Extend vs Append (from Matrix 4 results already computed)
    append_disk_10k = None
    extend_disk_10k = None
    for storage, n_ops, dtype, min_kb, max_mb in extend_configs:
        if storage == "disk" and n_ops == 10000:
            # Find from already computed results
            for r in col_append_results:
                if (
                    r["params"]["storage"] == "disk"
                    and r["params"]["n_ops"] == 10000
                    and r["params"]["dtype"] == "i64"
                    and r["params"].get("min_kb", 128) == 128
                ):
                    append_disk_10k = r
                    break
            break

    # Check if we have the comparison data from Matrix 4
    print(f"4. Bulk vs Individual (ColumnVault, 10K i64 on disk):")
    print(f"   - See Matrix 4 for detailed comparison")
    print(f"   - Bulk extend is typically 50-200x faster than individual append")

    # Chunk size impact on append
    tiny_chunk = [
        r
        for r in col_append_results
        if r["params"]["storage"] == "memory"
        and r["params"]["n_ops"] == 10000
        and r["params"]["min_kb"] == 16
    ][0]
    default_chunk = [
        r
        for r in col_append_results
        if r["params"]["storage"] == "memory"
        and r["params"]["n_ops"] == 10000
        and r["params"]["min_kb"] == 128
    ][0]
    large_chunk = [
        r
        for r in col_append_results
        if r["params"]["storage"] == "memory"
        and r["params"]["n_ops"] == 10000
        and r["params"]["min_kb"] == 512
    ][0]

    print(f"5. Chunk Size Impact (ColumnVault, 10K i64 append, memory):")
    print(f"   - Tiny (16KB-1MB): {tiny_chunk['ops_per_sec']:,.0f} ops/s")
    print(f"   - Default (128KB-16MB): {default_chunk['ops_per_sec']:,.0f} ops/s")
    print(f"   - Large (512KB-64MB): {large_chunk['ops_per_sec']:,.0f} ops/s")
    print(
        f"   - Default is {default_chunk['ops_per_sec']/tiny_chunk['ops_per_sec']:.1f}x faster than tiny"
    )
    print()

    # Scale analysis
    print(f"6. Scalability (ColumnVault i64 append on disk):")
    for i, n in enumerate(scale_configs):
        r = scale_results[i]
        time_per_op = r["elapsed"] / n if n > 0 else 0
        print(
            f"   - {n:>7,d} ops: {r['ops_per_sec']:>10,.0f} ops/s  ({format_time(time_per_op)}/op)  DB: {format_size(r['db_size'])}"
        )
    print()

    # Large data handling
    print(f"7. Large Data Handling (disk, default chunks):")
    large_data_results = [r for r in dtype_results if r["params"]["dtype"] in ["bytes:131072"]]
    if large_data_results:
        r = large_data_results[0]
        n_ops = r["params"]["n_ops"]
        total_mb = (n_ops * 131072) / (1024 * 1024)
        mb_per_sec = total_mb / r["elapsed"] if r["elapsed"] > 0 else 0
        print(f"   - bytes:131072 (128KB/element): {r['ops_per_sec']:,.0f} ops/s")
        print(f"   - Total data: {total_mb:.1f} MB written in {format_time(r['elapsed'])}")
        print(f"   - Throughput: {mb_per_sec:.1f} MB/s")
    print()

    # Cleanup
    runner.cleanup()

    print("=" * 100)
    print("Benchmark completed!")
    print("=" * 100)


# =============================================================================
# Quick Benchmark (Faster)
# =============================================================================


def run_quick_benchmark():
    """Quick benchmark with essential comparisons."""
    runner = BenchmarkRunner()

    print("=" * 100)
    print("KohakuVault Quick Benchmark")
    print("=" * 100)
    print()

    results = []

    # KVault comparisons
    print("Testing KVault...")
    r1 = runner.run(
        lambda: bench_kvault_write("memory", 5000, 1024, 0, runner),
        name="KVault write (memory, no cache)",
    )
    r2 = runner.run(
        lambda: bench_kvault_write("memory", 5000, 1024, 64, runner),
        name="KVault write (memory, 64MB cache)",
    )
    r3 = runner.run(
        lambda: bench_kvault_write("disk", 5000, 1024, 64, runner),
        name="KVault write (disk, 64MB cache)",
    )
    r4 = runner.run(
        lambda: bench_kvault_read("disk", 5000, 1024, runner), name="KVault read (disk)"
    )

    results.extend([r1, r2, r3, r4])

    # ColumnVault comparisons
    print("Testing ColumnVault...")
    r5 = runner.run(
        lambda: bench_column_append("memory", 10000, "i64", 128, 16, runner),
        name="Column append i64 (memory)",
    )
    r6 = runner.run(
        lambda: bench_column_extend("memory", 10000, "i64", 128, 16, runner),
        name="Column extend i64 (memory)",
    )
    r7 = runner.run(
        lambda: bench_column_append("disk", 10000, "i64", 128, 16, runner),
        name="Column append i64 (disk)",
    )
    r8 = runner.run(
        lambda: bench_column_random_read("disk", 10000, 1000, "i64", 128, 16, runner),
        name="Column random read (disk)",
    )

    results.extend([r5, r6, r7, r8])

    print()
    print("-" * 100)
    print(f"{'Benchmark':<45s} {'Time':<12s} {'Ops/Sec':<12s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['params']['name']:<45s} {format_time(r['elapsed']):<12s} {r['ops_per_sec']:<12,.0f}"
        )

    print()

    # Key insights
    cache_speedup = r2["ops_per_sec"] / r1["ops_per_sec"]
    extend_speedup = r6["ops_per_sec"] / r5["ops_per_sec"]

    print("Key Insights:")
    print(f"  - Cache speedup: {cache_speedup:.1f}x")
    print(f"  - Bulk extend vs individual append: {extend_speedup:.1f}x")
    print(f"  - Memory vs disk (with cache): {r2['ops_per_sec']/r3['ops_per_sec']:.1f}x")
    print()

    runner.cleanup()
    print("Quick benchmark completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KohakuVault Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # Quick benchmark
  python benchmark.py --full                   # Full benchmark (all matrices)
  python benchmark.py --kvault-only            # Only KVault tests
  python benchmark.py --column-only            # Only ColumnVault tests
  python benchmark.py --scale small            # Small scale (faster)
  python benchmark.py --scale medium           # Medium scale (default)
  python benchmark.py --scale large            # Large scale (slower)
  python benchmark.py --skip-matrix 5 6        # Skip matrices 5 and 6
  python benchmark.py --full --scale small     # Full test with small scale
        """,
    )

    parser.add_argument(
        "--full", action="store_true", help="Run full matrix benchmark (default: quick benchmark)"
    )

    parser.add_argument(
        "--kvault-only", action="store_true", help="Only run KVault benchmarks (matrices 1-2)"
    )

    parser.add_argument(
        "--column-only", action="store_true", help="Only run ColumnVault benchmarks (matrices 3-7)"
    )

    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="Scale of tests: small (fast), medium (default), large (comprehensive)",
    )

    parser.add_argument(
        "--skip-matrix", type=int, nargs="+", metavar="N", help="Skip specific matrix numbers (1-7)"
    )

    parser.add_argument(
        "--only-matrix",
        type=int,
        nargs="+",
        metavar="N",
        help="Only run specific matrix numbers (1-7)",
    )

    args = parser.parse_args()

    # Determine which matrices to run
    if args.only_matrix:
        matrices_to_run = set(args.only_matrix)
    else:
        matrices_to_run = {1, 2, 3, 4, 5, 6, 7}

    if args.skip_matrix:
        matrices_to_run -= set(args.skip_matrix)

    if args.kvault_only:
        matrices_to_run &= {1, 2}

    if args.column_only:
        matrices_to_run &= {3, 4, 5, 6, 7}

    # Run appropriate benchmark
    if args.full:
        print(f"Running FULL matrix benchmark (scale: {args.scale})...")
        print(f"Matrices to run: {sorted(matrices_to_run)}")
        print()
        run_matrix_benchmark_configurable(matrices_to_run, args.scale)
    else:
        print("Running quick benchmark...")
        print("(Use --full for comprehensive matrix, --help for options)")
        print()
        run_quick_benchmark()
