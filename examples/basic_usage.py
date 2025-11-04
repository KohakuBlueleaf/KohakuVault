"""
Basic KohakuVault Usage Examples

Demonstrates key-value storage with streaming and caching.
"""

import time
from kohakuvault import KVault

# =============================================================================
# Example 1: Basic Key-Value Operations
# =============================================================================

print("Example 1: Basic Key-Value Storage")
print("=" * 50)

vault = KVault("data.db")

# Store binary data
vault["user:123:avatar"] = b"\x89PNG\r\n\x1a\n..."  # PNG bytes
vault["config:api_key"] = b"secret-key-here"

# Retrieve
avatar = vault["user:123:avatar"]
print(f"Stored avatar: {len(avatar)} bytes")

# Check existence
if "config:api_key" in vault:
    print("API key exists")

# Delete
del vault["user:123:avatar"]
print(f"Keys remaining: {len(vault)}")

vault.close()
print()

# =============================================================================
# Example 2: Streaming Large Files
# =============================================================================

print("Example 2: Streaming Large Files")
print("=" * 50)

vault = KVault("media.db", chunk_size=1024 * 1024)  # 1 MiB chunks

# Create a test file
with open("test_video.dat", "wb") as f:
    f.write(b"x" * 10_000_000)  # 10 MB fake video

# Stream from file to vault (doesn't load entire file into RAM)
with open("test_video.dat", "rb") as f:
    vault.put_file("video:demo", f)

print(f"Stored video: {len(vault['video:demo'])} bytes")

# Stream from vault to file
with open("downloaded_video.dat", "wb") as f:
    vault.get_to_file("video:demo", f)

print("Video downloaded")

# Cleanup
import os

os.remove("test_video.dat")
os.remove("downloaded_video.dat")
vault.close()
print()

# =============================================================================
# Example 3: Write-Back Caching for Bulk Operations
# =============================================================================

print("Example 3: Write-Back Caching (Context Manager - Recommended)")
print("=" * 50)

vault = KVault("thumbnails.db")

# Context manager automatically handles flushing
print("Writing 1000 thumbnails with automatic cache...")
with vault.cache(cap_bytes=64 * 1024 * 1024):
    for i in range(1000):
        vault[f"thumb:{i}"] = b"fake_thumbnail_data_" + str(i).encode()
# Auto-flushed here!

print(f"Total keys: {len(vault)}")
vault.close()
print()

# =============================================================================
# Example 3b: Long-Running with Daemon Thread
# =============================================================================

print("Example 3b: Daemon Thread Auto-Flush")
print("=" * 50)

vault2 = KVault("sensors.db")

# Enable cache with daemon thread (flushes every 2 seconds)
vault2.enable_cache(
    cap_bytes=64 * 1024 * 1024,
    flush_threshold=16 * 1024 * 1024,
    flush_interval=2.0,  # Auto-flush every 2 seconds
)

print("Writing sensor data with daemon auto-flush...")
for i in range(100):
    vault2[f"sensor:{i}"] = b"reading_data_" + str(i).encode()
    if i % 20 == 0:
        time.sleep(0.1)  # Simulate real-world timing

print("Data written, daemon will flush automatically")
time.sleep(2.5)  # Wait for daemon flush
print(f"Total keys: {len(vault2)}")

vault2.close()  # Stops daemon and flushes remaining data
print()

# =============================================================================
# Example 4: Context Manager
# =============================================================================

print("Example 4: Context Manager")
print("=" * 50)

# Auto-flushes cache and closes on exit
with KVault("temp.db") as vault:
    vault.enable_cache()

    vault["key1"] = b"value1"
    vault["key2"] = b"value2"

    print(f"Stored {len(vault)} keys")
    # Cache auto-flushed on exit

print("Vault auto-closed")
print()

# =============================================================================
# Example 5: Custom Configuration
# =============================================================================

print("Example 5: Custom Configuration")
print("=" * 50)

vault = KVault(
    path="optimized.db",
    chunk_size=2 * 1024 * 1024,  # 2 MiB streaming chunks
    retries=10,  # More retries for busy DB
    backoff_base=0.05,  # 50ms initial backoff
    enable_wal=True,  # Write-Ahead Logging (default)
    cache_kb=20000,  # 20 MB SQLite cache
)

vault["test"] = b"data"
print("Configured vault created")

vault.close()
print()

# Cleanup test databases
for db in ["data.db", "media.db", "thumbnails.db", "sensors.db", "temp.db", "optimized.db"]:
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"{db}{ext}")
        except FileNotFoundError:
            pass

print("All examples completed!")
