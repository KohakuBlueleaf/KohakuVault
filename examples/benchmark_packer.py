"""
DataPacker Benchmark Script

Compare Python struct.pack vs Rust DataPacker performance.
Run this to measure actual performance improvements on your system.
"""

import time
import struct
import json
from kohakuvault import DataPacker, ColumnVault


def benchmark_python_packing(n=10000):
    """Benchmark: Old way using Python struct.pack"""
    print(f"\n1. Python struct.pack (i64, {n} items)")
    print("-" * 60)

    start = time.time()
    for i in range(n):
        packed = struct.pack("<q", i)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n/elapsed:.0f} ops/s")
    return elapsed


def benchmark_rust_packing(n=10000):
    """Benchmark: New way using Rust DataPacker"""
    print(f"\n2. Rust DataPacker.pack() (i64, {n} items)")
    print("-" * 60)

    packer = DataPacker("i64")
    start = time.time()
    for i in range(n):
        packed = packer.pack(i)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n/elapsed:.0f} ops/s")
    return elapsed


def benchmark_rust_bulk_packing(n=10000):
    """Benchmark: Bulk packing with pack_many"""
    print(f"\n3. Rust DataPacker.pack_many() (i64, {n} items)")
    print("-" * 60)

    packer = DataPacker("i64")
    values = list(range(n))

    start = time.time()
    packed = packer.pack_many(values)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n/elapsed:.0f} ops/s")
    return elapsed


def benchmark_messagepack_vs_json(n=1000):
    """Benchmark: MessagePack vs JSON for structured data"""
    print(f"\n4. MessagePack vs JSON ({n} items)")
    print("-" * 60)

    data = {
        "name": "Test User",
        "age": 30,
        "scores": [95, 87, 92, 88, 91],
        "active": True,
        "metadata": {"created": "2025-01-01", "verified": True},
    }

    # Python JSON approach
    start = time.time()
    for _ in range(n):
        json_str = json.dumps(data)
        encoded = json_str.encode("utf-8")
    json_elapsed = time.time() - start
    json_size = len(json_str.encode("utf-8"))

    # Rust MessagePack
    packer = DataPacker("msgpack")
    start = time.time()
    for _ in range(n):
        packed = packer.pack(data)
    msgpack_elapsed = time.time() - start
    msgpack_size = len(packer.pack(data))

    print(f"  JSON:")
    print(f"    Time: {json_elapsed:.3f}s")
    print(f"    Throughput: {n/json_elapsed:.0f} ops/s")
    print(f"    Size: {json_size} bytes")

    print(f"  MessagePack:")
    print(f"    Time: {msgpack_elapsed:.3f}s")
    print(f"    Throughput: {n/msgpack_elapsed:.0f} ops/s")
    print(f"    Size: {msgpack_size} bytes ({100*msgpack_size/json_size:.1f}% of JSON)")

    if json_elapsed > msgpack_elapsed:
        speedup = json_elapsed / msgpack_elapsed
        print(f"  Speedup: {speedup:.2f}x faster")
    else:
        slowdown = msgpack_elapsed / json_elapsed
        print(f"  Note: MessagePack {slowdown:.2f}x slower (unexpected)")


def benchmark_column_append(n=10000):
    """Benchmark: Column append with Rust vs Python packing"""
    print(f"\n5. Column.append() with Rust packer ({n} items)")
    print("-" * 60)

    vault = ColumnVault(":memory:")
    col = vault.create_column("test_rust", "i64", use_rust_packer=True)

    start = time.time()
    for i in range(n):
        col.append(i)
    rust_elapsed = time.time() - start

    print(f"  Rust packer:")
    print(f"    Time: {rust_elapsed:.3f}s")
    print(f"    Throughput: {n/rust_elapsed:.0f} ops/s")

    print(f"\n6. Column.append() with Python packer ({n} items)")
    print("-" * 60)

    vault2 = ColumnVault(":memory:")
    col2 = vault2.create_column("test_python", "i64", use_rust_packer=False)

    start = time.time()
    for i in range(n):
        col2.append(i)
    python_elapsed = time.time() - start

    print(f"  Python packer:")
    print(f"    Time: {python_elapsed:.3f}s")
    print(f"    Throughput: {n/python_elapsed:.0f} ops/s")

    if python_elapsed > rust_elapsed:
        speedup = python_elapsed / rust_elapsed
        print(f"  Speedup: {speedup:.2f}x faster with Rust packer")
    else:
        print(f"  Note: Results similar or Python faster (unexpected)")


def benchmark_column_extend(n=10000):
    """Benchmark: Column extend (bulk operation)"""
    print(f"\n7. Column.extend() with Rust packer ({n} items)")
    print("-" * 60)

    vault = ColumnVault(":memory:")
    col = vault.create_column("test", "i64")
    values = list(range(n))

    start = time.time()
    col.extend(values)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n/elapsed:.0f} ops/s")


def benchmark_string_encodings(n=1000):
    """Benchmark: Different string encodings"""
    print(f"\n8. String Encoding Comparison ({n} items)")
    print("-" * 60)

    text = "Hello, World! This is a test string."

    encodings = [
        ("UTF-8", "str:utf8"),
        ("UTF-16LE", "str:utf16le"),
        ("ASCII", "str:ascii"),
    ]

    for name, dtype in encodings:
        packer = DataPacker(dtype)

        start = time.time()
        for _ in range(n):
            packed = packer.pack(text if name != "ASCII" else "Hello World")
        elapsed = time.time() - start

        sample_packed = packer.pack(text if name != "ASCII" else "Hello World")
        print(f"  {name}:")
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Throughput: {n/elapsed:.0f} ops/s")
        print(f"    Size: {len(sample_packed)} bytes")


def benchmark_fixed_vs_variable_size(n=1000):
    """Benchmark: Fixed-size vs variable-size types"""
    print(f"\n9. Fixed vs Variable Size ({n} items)")
    print("-" * 60)

    # Fixed-size string (32 bytes)
    packer_fixed = DataPacker("str:32:utf8")
    text = "short"

    start = time.time()
    for _ in range(n):
        packed = packer_fixed.pack(text)
    fixed_elapsed = time.time() - start

    print(f"  Fixed-size (str:32:utf8):")
    print(f"    Time: {fixed_elapsed:.3f}s")
    print(f"    Throughput: {n/fixed_elapsed:.0f} ops/s")
    print(f"    Size: {len(packer_fixed.pack(text))} bytes (padded)")

    # Variable-size string
    packer_var = DataPacker("str:utf8")

    start = time.time()
    for _ in range(n):
        packed = packer_var.pack(text)
    var_elapsed = time.time() - start

    print(f"  Variable-size (str:utf8):")
    print(f"    Time: {var_elapsed:.3f}s")
    print(f"    Throughput: {n/var_elapsed:.0f} ops/s")
    print(f"    Size: {len(packer_var.pack(text))} bytes (exact)")


def main():
    """Run all benchmarks"""
    print("=" * 60)
    print("DataPacker Performance Benchmarks")
    print("=" * 60)
    print("\nMeasuring actual performance on your system...")
    print("(Numbers will vary based on CPU, OS, Python version)")

    try:
        # Basic packing benchmarks
        python_time = benchmark_python_packing(n=10000)
        rust_time = benchmark_rust_packing(n=10000)
        bulk_time = benchmark_rust_bulk_packing(n=10000)

        print(f"\n" + "=" * 60)
        print("Summary (Primitive Packing)")
        print("=" * 60)
        if rust_time < python_time:
            print(f"Rust vs Python: {python_time/rust_time:.2f}x faster")
        else:
            print(f"Rust vs Python: Similar performance")

        if bulk_time < rust_time:
            print(f"Bulk vs Single: {rust_time/bulk_time:.2f}x faster")

        # MessagePack benchmark
        benchmark_messagepack_vs_json(n=1000)

        # Column integration benchmarks
        benchmark_column_append(n=5000)  # Reduced n for append
        benchmark_column_extend(n=10000)

        # String encoding benchmarks
        benchmark_string_encodings(n=1000)

        # Fixed vs variable size
        benchmark_fixed_vs_variable_size(n=1000)

        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Bulk operations (extend/pack_many) are significantly faster")
        print("2. MessagePack is more compact than JSON")
        print("3. Use Rust packer for best performance (default in Column)")
        print("4. Fixed-size types allow batch operations (pack_many/unpack_many)")

    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
