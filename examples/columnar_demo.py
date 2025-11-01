"""
Columnar Storage Demo

Demonstrates list-like columnar storage for structured data.
"""

from kohakuvault import KVault, ColumnVault
import time

# =============================================================================
# Example 1: Time-Series Data
# =============================================================================

print("Example 1: Time-Series Sensor Data")
print("=" * 60)

kv = KVault("sensors.db")
cv = ColumnVault(kv)

# Create columns for sensor readings
cv.create_column("timestamps", "i64")
cv.create_column("temperatures", "f64")
cv.create_column("humidity", "f64")

# Get column references
timestamps = cv["timestamps"]
temperatures = cv["temperatures"]
humidity = cv["humidity"]

# Simulate sensor readings
print("Recording sensor data...")
for i in range(100):
    timestamps.append(int(time.time()) + i)
    temperatures.append(20.0 + (i % 10))
    humidity.append(60.0 + (i % 20))

print(f"Recorded {len(timestamps)} readings")
print(f"First reading: {timestamps[0]}, {temperatures[0]}°C, {humidity[0]}%")
print(f"Last reading: {timestamps[-1]}, {temperatures[-1]}°C, {humidity[-1]}%")

# Meanwhile, use KV store for metadata
kv["sensor:location"] = b"warehouse-floor-2"
kv["sensor:id"] = b"TEMP-001"

print()

# =============================================================================
# Example 2: Log Storage with Variable-Size Strings
# =============================================================================

print("Example 2: Application Logs")
print("=" * 60)

cv.create_column("log_timestamps", "i64")
cv.create_column("log_messages", "bytes")  # Variable-size!

logs_ts = cv["log_timestamps"]
logs_msg = cv["log_messages"]

# Store log entries
log_entries = [
    (1234567890, b"Server started"),
    (1234567891, b"Database connection established"),
    (1234567892, b"Processing request from 192.168.1.100"),
    (1234567893, b"ERROR: Failed to connect to external API: Connection timeout after 30s"),
    (1234567894, b"Retrying..."),
    (1234567895, b"Success"),
]

print("Storing log entries...")
for ts, msg in log_entries:
    logs_ts.append(ts)
    logs_msg.append(msg)

print(f"Stored {len(logs_msg)} log entries")

# Query logs
print("\nLog entries:")
for i, (ts, msg) in enumerate(zip(logs_ts, logs_msg)):
    msg_str = msg.decode("utf-8")
    print(f"  [{i}] {ts}: {msg_str}")

print()

# =============================================================================
# Example 3: Feature Vectors for ML
# =============================================================================

print("Example 3: ML Feature Vectors")
print("=" * 60)

cv.create_column("user_ids", "i64")
cv.create_column("embeddings", "bytes:512")  # Fixed-size 512-byte vectors

user_ids = cv["user_ids"]
embeddings = cv["embeddings"]

# Store embeddings
print("Storing user embeddings...")
for user_id in range(50):
    user_ids.append(user_id)
    # Fake embedding (normally from neural network)
    fake_embedding = bytes([user_id % 256] * 512)
    embeddings.append(fake_embedding)

print(f"Stored {len(embeddings)} embeddings")
print(f"User 0 embedding: {embeddings[0][:10]}... ({len(embeddings[0])} bytes)")
print(f"User 25 embedding: {embeddings[25][:10]}... ({len(embeddings[25])} bytes)")

print()

# =============================================================================
# Example 4: Mixed Fixed and Variable Types
# =============================================================================

print("Example 4: User Activity Tracking")
print("=" * 60)

cv.create_column("activity_user_id", "i64")
cv.create_column("activity_timestamp", "i64")
cv.create_column("activity_type", "bytes:32")  # Fixed: "login", "logout", etc.
cv.create_column("activity_details", "bytes")  # Variable: JSON or description

act_user = cv["activity_user_id"]
act_time = cv["activity_timestamp"]
act_type = cv["activity_type"]
act_details = cv["activity_details"]

# Record activities
activities = [
    (123, 1234567900, b"login", b"IP: 192.168.1.50"),
    (123, 1234567950, b"view_page", b"Page: /dashboard, Duration: 5.2s"),
    (456, 1234568000, b"login", b"IP: 10.0.0.100, User-Agent: Chrome/120"),
    (123, 1234568100, b"logout", b"Session duration: 200s"),
]

for user_id, ts, act, details in activities:
    act_user.append(user_id)
    act_time.append(ts)
    act_type.append(act)
    act_details.append(details)

print(f"Tracked {len(act_user)} activities")

# Query specific user
print("\nUser 123 activities:")
for i in range(len(act_user)):
    if act_user[i] == 123:
        act_str = act_type[i].rstrip(b"\x00").decode()
        details_str = act_details[i].decode()
        print(f"  {act_time[i]}: {act_str} - {details_str}")

print()

# =============================================================================
# Example 5: Incremental Chunk Growth Demo
# =============================================================================

print("Example 5: Incremental Chunk Growth")
print("=" * 60)

# Create column with custom chunk sizing
cv2 = ColumnVault(
    "growth_demo.db",
    min_chunk_bytes=128 * 1024,  # Start at 128KB
    max_chunk_bytes=16 * 1024 * 1024,  # Max 16MB
)

cv2.create_column("growth_test", "i64")
col = cv2["growth_test"]

# Add lots of data to demonstrate chunk growth
print("Appending 100,000 int64 values...")
for i in range(100_000):
    col.append(i)

print(f"Stored {len(col)} values")
print(f"First value: {col[0]}")
print(f"Last value: {col[-1]}")
print("\nChunk growth pattern: 128KB → 256KB → 512KB → 1MB → 2MB → 4MB → 8MB → 16MB")
print("(Each chunk doubles in size until reaching 16MB max)")

print()

# =============================================================================
# Cleanup
# =============================================================================

kv.close()

import os

for db in ["sensors.db", "growth_demo.db"]:
    for ext in ["", "-wal", "-shm"]:
        try:
            os.remove(f"{db}{ext}")
        except (FileNotFoundError, PermissionError):
            pass

print("All columnar examples completed!")
