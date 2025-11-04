"""
DataPacker Demo

Demonstrates the Rust-based DataPacker for efficient data serialization.
"""

from kohakuvault import DataPacker, ColumnVault
import json


def demo_primitives():
    """Demo: Primitive types (i64, f64, strings, bytes)"""
    print("=" * 60)
    print("1. Primitive Types Demo")
    print("=" * 60)

    # Integer packing
    print("\n[i64 Packing]")
    packer = DataPacker("i64")
    packed = packer.pack(42)
    print(f"  Input: 42")
    print(f"  Packed: {len(packed)} bytes")
    print(f"  Unpacked: {packer.unpack(packed, 0)}")

    # Float packing
    print("\n[f64 Packing]")
    packer = DataPacker("f64")
    packed = packer.pack(3.14159)
    print(f"  Input: 3.14159")
    print(f"  Packed: {len(packed)} bytes")
    print(f"  Unpacked: {packer.unpack(packed, 0)}")

    # String packing (UTF-8)
    print("\n[String UTF-8 Packing]")
    packer = DataPacker("str:utf8")
    text = "Hello, 世界!"
    packed = packer.pack(text)
    print(f"  Input: {text}")
    print(f"  Packed: {len(packed)} bytes")
    print(f"  Unpacked: {packer.unpack(packed, 0)}")

    # Fixed-size bytes
    print("\n[Fixed-size Bytes (128 bytes)]")
    packer = DataPacker("bytes:128")
    data = b"test data"
    packed = packer.pack(data)
    print(f"  Input: {len(data)} bytes")
    print(f"  Packed: {len(packed)} bytes (padded to 128)")
    print(f"  Unpacked (first 20 bytes): {packer.unpack(packed, 0)[:20]}")


def demo_messagepack():
    """Demo: MessagePack for structured data"""
    print("\n" + "=" * 60)
    print("2. MessagePack Demo (Structured Data)")
    print("=" * 60)

    packer = DataPacker("msgpack")

    # Complex nested structure
    user = {
        "user_id": 12345,
        "profile": {
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 30,
            "verified": True,
        },
        "permissions": ["read", "write", "admin"],
        "metadata": {
            "created_at": "2025-01-01T00:00:00Z",
            "last_login": "2025-01-10T15:30:00Z",
            "login_count": 42,
        },
    }

    # Pack with MessagePack
    msgpack_packed = packer.pack(user)

    # Compare with JSON
    json_str = json.dumps(user)
    json_bytes = json_str.encode("utf-8")

    print(f"\n[Size Comparison]")
    print(f"  JSON: {len(json_bytes)} bytes")
    print(f"  MessagePack: {len(msgpack_packed)} bytes")
    print(f"  Compression: {100 * len(msgpack_packed) / len(json_bytes):.1f}%")

    # Verify roundtrip
    unpacked = packer.unpack(msgpack_packed, 0)
    print(f"\n[Roundtrip Verification]")
    print(f"  Original == Unpacked: {user == unpacked}")


def demo_json_schema_validation():
    """Demo: JSON Schema validation"""
    print("\n" + "=" * 60)
    print("3. JSON Schema Validation Demo")
    print("=" * 60)

    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "minLength": 3,
                "maxLength": 20,
                "pattern": "^[a-zA-Z0-9_]+$",
            },
            "email": {"type": "string", "format": "email"},
            "age": {"type": "integer", "minimum": 13, "maximum": 120},
        },
        "required": ["username", "email"],
    }

    packer = DataPacker.with_json_schema(schema)

    # Valid user
    print("\n[Valid Data]")
    valid_user = {"username": "alice123", "email": "alice@example.com", "age": 30}
    packed = packer.pack(valid_user)
    print(f"  Input: {valid_user}")
    print(f"  Packed: {len(packed)} bytes")
    print(f"  Success!")

    # Invalid user (missing email)
    print("\n[Invalid Data - Missing Required Field]")
    try:
        invalid_user = {"username": "bob"}
        packer.pack(invalid_user)
        print("  ERROR: Should have failed validation!")
    except ValueError as e:
        print(f"  Validation failed (expected): {str(e)[:80]}...")

    # Invalid user (age out of range)
    print("\n[Invalid Data - Age Out of Range]")
    try:
        invalid_user = {"username": "charlie", "email": "charlie@example.com", "age": 10}
        packer.pack(invalid_user)
        print("  ERROR: Should have failed validation!")
    except ValueError as e:
        print(f"  Validation failed (expected): {str(e)[:80]}...")


def demo_column_integration():
    """Demo: DataPacker integration with ColumnVault"""
    print("\n" + "=" * 60)
    print("4. Column Integration Demo")
    print("=" * 60)

    vault = ColumnVault(":memory:")

    # Create columns with different types
    print("\n[Creating Columns]")
    ages = vault.create_column("ages", "i64")
    scores = vault.create_column("scores", "f64")
    ids = vault.create_column("ids", "bytes:16")  # 16-byte fixed IDs

    print("  Created: ages (i64), scores (f64), ids (bytes:16)")

    # Add data
    print("\n[Adding Data with Rust Packer]")
    ages.extend([25, 30, 35, 40, 45])
    scores.extend([85.5, 92.3, 78.9, 88.1, 95.7])
    ids.extend([b"alice", b"bob", b"charlie", b"david", b"eve"])

    print(f"  Added {len(ages)} entries")

    # Retrieve data
    print("\n[Reading Data]")
    for i in range(len(ages)):
        id_bytes = ids[i]
        name = id_bytes.rstrip(b"\x00").decode("utf-8")
        print(f"  {name}: age={ages[i]}, score={scores[i]:.1f}")


def demo_bulk_operations():
    """Demo: Bulk operations performance"""
    print("\n" + "=" * 60)
    print("5. Bulk Operations Demo")
    print("=" * 60)

    import time

    vault = ColumnVault(":memory:")
    col = vault.create_column("numbers", "i64")

    n = 10000

    # Test extend (bulk operation)
    print(f"\n[Bulk Extend - {n} items]")
    values = list(range(n))
    start = time.time()
    col.extend(values)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n/elapsed:.0f} ops/s")
    print(f"  Verification: sum = {sum(col)}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("KohakuVault DataPacker Demonstration")
    print("=" * 60)

    try:
        demo_primitives()
        demo_messagepack()
        demo_json_schema_validation()
        demo_column_integration()
        demo_bulk_operations()

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nFor more examples, see:")
        print("  - docs/datapacker.md (comprehensive documentation)")
        print("  - examples/benchmark_packer.py (performance benchmarks)")
        print("  - tests/test_packer.py (unit tests)")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
