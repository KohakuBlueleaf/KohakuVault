from kohakuvault import KVault

v = KVault(":memory:")

# Test missing key raises KeyError
try:
    result = v["missing"]
    print(f"ERROR: Should have raised KeyError, got: {result!r}")
except KeyError as e:
    print(f"SUCCESS: Got KeyError as expected: {e}")

# Test get with default
result = v.get("missing", b"default")
print(f"get with default: {result!r}")

# Test get without default
result = v.get("missing")
print(f"get without default: {result!r}")

v.close()
