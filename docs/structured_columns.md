# Structured Data Columns

## Overview

KohakuVault now supports **structured data types** in columnar storage through the DataPacker integration. You can now store dictionaries, lists, and strings directly in columns without manual serialization.

## Supported Structured Types

### MessagePack Columns (`msgpack`)

Store Python dicts, lists, and nested structures in a compact binary format.

```python
from kohakuvault import ColumnVault

vault = ColumnVault("data.db")
users = vault.create_column("users", "msgpack")

# Store dictionaries
users.append({"name": "Alice", "age": 30, "tags": ["vip", "verified"]})
users.append({"name": "Bob", "age": 25, "active": True})

# Retrieve
user = users[0]
print(user["name"])  # "Alice"

# Bulk operations
records = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
users.extend(records)
```

### CBOR Columns (`cbor`)

Alternative IETF standard format for structured data.

```python
vault = ColumnVault("data.db")
events = vault.create_column("events", "cbor")

events.append({"type": "login", "user_id": 123, "timestamp": 1234567890})
events.append({"type": "logout", "user_id": 123})

event = events[0]
print(event["type"])  # "login"
```

### String Columns

Store variable-length strings with different encodings.

#### UTF-8 Strings (`str:utf8`)

```python
messages = vault.create_column("messages", "str:utf8")

messages.append("Hello, World!")
messages.append("Unicode: 世界 🌍")
messages.append("Variable length strings...")

print(messages[1])  # "Unicode: 世界 🌍"
```

#### ASCII Strings (`str:ascii`)

```python
ascii_data = vault.create_column("ascii_only", "str:ascii")

ascii_data.append("ASCII text only")
# ascii_data.append("世界")  # ERROR: non-ASCII characters
```

#### Other Encodings

- `str:utf16le` - UTF-16 Little Endian (Windows)
- `str:utf16be` - UTF-16 Big Endian
- `str:latin1` - Latin-1 / ISO-8859-1

```python
utf16_col = vault.create_column("utf16_data", "str:utf16le")
utf16_col.append("Windows compatible text")

latin1_col = vault.create_column("latin1_text", "str:latin1")
latin1_col.append("Café")
```

## How It Works

### Storage Architecture

**Variable-size columns** (msgpack, cbor, variable strings) use a two-column approach:

```
Column "users" (dtype="msgpack")
  ├─> users_data: Stores MessagePack bytes (dtype stored as "msgpack")
  └─> users_idx:  Stores prefix-sum offsets (i64)
```

When you create a column with `dtype="msgpack"`:
1. Creates `{name}_data` column with dtype="msgpack" in metadata
2. Creates `{name}_idx` column with dtype="i64"
3. DataPacker is created from dtype string automatically

**On reload:**
1. Detects `users_data` and `users_idx` exist
2. Reads dtype="msgpack" from `users_data` metadata
3. Creates `DataPacker("msgpack")`
4. Automatically unpacks data when accessed

### Fixed vs Variable Size Detection

The `parse_dtype()` function uses DataPacker to automatically determine if a type is fixed or variable-size:

```python
# Automatically detected as fixed-size (8 bytes)
col = vault.create_column("ages", "i64")

# Automatically detected as fixed-size (32 bytes)
col = vault.create_column("codes", "str:32:utf8")

# Automatically detected as variable-size
col = vault.create_column("messages", "str:utf8")

# Automatically detected as variable-size
col = vault.create_column("data", "msgpack")
```

## Examples

### Example 1: User Profiles with MessagePack

```python
from kohakuvault import ColumnVault

vault = ColumnVault("profiles.db")
profiles = vault.create_column("user_profiles", "msgpack")

# Add user profiles
profiles.append({
    "user_id": 12345,
    "profile": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "bio": "Software engineer"
    },
    "settings": {
        "theme": "dark",
        "notifications": True
    },
    "tags": ["premium", "verified"]
})

# Retrieve and use
user = profiles[0]
print(f"{user['profile']['name']} - {user['profile']['email']}")

# Query by iterating
for profile in profiles:
    if "premium" in profile.get("tags", []):
        print(f"Premium user: {profile['user_id']}")
```

### Example 2: Mixed Column Types

```python
vault = ColumnVault("app.db")

# Create columns of different types
user_ids = vault.create_column("user_ids", "i64")
scores = vault.create_column("scores", "f64")
usernames = vault.create_column("usernames", "str:utf8")
metadata = vault.create_column("metadata", "msgpack")

# Add data
for i in range(100):
    user_ids.append(i)
    scores.append(i * 1.5)
    usernames.append(f"user_{i}")
    metadata.append({
        "created": "2025-01-01",
        "active": True,
        "preferences": {"theme": "dark"}
    })

# Access data
print(f"User {usernames[42]}: score={scores[42]}, meta={metadata[42]}")
```

### Example 3: Event Logging with CBOR

```python
vault = ColumnVault("events.db")
events = vault.create_column("audit_log", "cbor")

# Log events
events.append({
    "timestamp": "2025-01-01T12:00:00Z",
    "event_type": "login",
    "user_id": 123,
    "ip_address": "192.168.1.1"
})

events.append({
    "timestamp": "2025-01-01T12:05:00Z",
    "event_type": "file_upload",
    "user_id": 123,
    "file_size": 1048576,
    "file_name": "document.pdf"
})

# Query events
for event in events:
    if event["event_type"] == "login":
        print(f"Login from {event['ip_address']}")
```

### Example 4: Multilingual Text Storage

```python
vault = ColumnVault("i18n.db")

# Different language columns
english = vault.create_column("text_en", "str:utf8")
chinese = vault.create_column("text_zh", "str:utf8")
arabic = vault.create_column("text_ar", "str:utf8")

# Add translations
english.append("Hello, World!")
chinese.append("你好，世界！")
arabic.append("مرحبا بالعالم")

# All stored efficiently with proper encoding
print(f"EN: {english[0]}")
print(f"ZH: {chinese[0]}")
print(f"AR: {arabic[0]}")
```

## Performance Characteristics

### MessagePack Performance

- **Size**: ~59-73% of JSON (measured)
- **Append**: Comparable to raw bytes
- **Bulk extend**: ~3M ops/s
- **Retrieval**: Fast deserialization

### String Performance

- **UTF-8**: Native Python string encoding, fastest
- **UTF-16**: 2x size, slower encoding/decoding
- **ASCII**: Validates only 7-bit characters, fast
- **Latin-1**: 1-byte per character, fast

## Persistence

All structured dtypes persist correctly:

```python
# Write data
vault1 = ColumnVault("data.db")
col1 = vault1.create_column("users", "msgpack")
col1.append({"name": "Alice", "age": 30})

# Reload later
vault2 = ColumnVault("data.db")
col2 = vault2["users"]  # Automatically reconstructs as msgpack column
user = col2[0]          # Automatically unpacks to dict
print(user["name"])     # "Alice"
```

The dtype is stored in the database metadata and automatically reconstructed on reload.

## Limitations

### Current Limitations

1. **No Schema Evolution**: Changing structure requires manual migration
2. **No Indexing**: Cannot index into structured data (must iterate)
3. **No Partial Updates**: Must read entire object, modify, write back
4. **Fixed-size structured not supported**: msgpack/cbor are always variable-size

### Workarounds

**Schema Evolution**: Use versioned dtypes
```python
# Version 1
v1_col = vault.create_column("data_v1", "msgpack")

# Version 2 (new schema)
v2_col = vault.create_column("data_v2", "msgpack")

# Migrate manually
for item in v1_col:
    migrated = migrate_v1_to_v2(item)
    v2_col.append(migrated)
```

**Indexing**: Use separate index columns
```python
ids = vault.create_column("user_ids", "i64")  # Indexed
data = vault.create_column("user_data", "msgpack")  # Full records

# Find by ID
target_id = 12345
for i, uid in enumerate(ids):
    if uid == target_id:
        user = data[i]
        break
```

## Best Practices

### 1. Choose the Right Format

- **MessagePack**: General purpose, best size efficiency
- **CBOR**: When you need IETF standard compliance
- **Strings**: When you need text search or language-specific encoding

### 2. Use Consistent Schemas

```python
# Good: Consistent structure
users.append({"name": "Alice", "age": 30})
users.append({"name": "Bob", "age": 25})

# OK but not ideal: Mixed structures
data.append({"type": "user", "name": "Alice"})
data.append({"type": "event", "timestamp": 123})
```

### 3. Validate with JSON Schema

```python
from kohakuvault import DataPacker

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name", "age"]
}

# Create validated packer
packer = DataPacker.with_json_schema(schema)

# Use in column (manual validation)
vault = ColumnVault("validated.db")
col = vault.create_column("users", "msgpack")

for user in user_list:
    # Validate before storing
    packed = packer.pack(user)  # Raises ValueError if invalid
    # Then store raw bytes
    # (Future: could integrate validation directly into Column)
```

### 4. Consider Fixed-size for Performance

When possible, use fixed-size types for better performance:

```python
# Variable-size (slower, uses prefix-sum index)
col1 = vault.create_column("messages", "str:utf8")

# Fixed-size (faster, direct indexing)
col2 = vault.create_column("codes", "str:32:utf8")  # Padded to 32 bytes
```

## Migration from Raw Bytes

If you were previously storing serialized data as raw bytes:

**Before:**
```python
import json

col = vault.create_column("data", "bytes")

# Manual serialization
data = {"name": "Alice"}
json_bytes = json.dumps(data).encode('utf-8')
col.append(json_bytes)

# Manual deserialization
retrieved_bytes = col[0]
retrieved_data = json.loads(retrieved_bytes.decode('utf-8'))
```

**After:**
```python
# Automatic serialization/deserialization
col = vault.create_column("data", "msgpack")

data = {"name": "Alice"}
col.append(data)  # Automatically serialized

retrieved_data = col[0]  # Automatically deserialized
print(retrieved_data["name"])  # "Alice"
```

**Benefits:**
- No manual JSON encoding/decoding
- Smaller size (MessagePack is more compact)
- Type preservation (ints stay ints, not floats)
- Cleaner code

## See Also

- [DataPacker API Reference](datapacker.md) - Full DataPacker documentation
- [Column Storage](col.md) - Column storage basics
- [MessagePack Specification](https://msgpack.org/)
- [CBOR RFC 8949](https://www.rfc-editor.org/rfc/rfc8949.html)
- [JSON Schema](https://json-schema.org/)
