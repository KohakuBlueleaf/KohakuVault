"""
Type wrappers for explicit encoding control in auto-packing mode.

These wrappers signal to KVault which encoding to use when auto-packing is enabled.
"""


class EncodingWrapper:
    """Base class for encoding type wrappers."""

    def __init__(self, data, encoding_name):
        self.data = data
        self.encoding_name = encoding_name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data!r})"


class MsgPack(EncodingWrapper):
    """
    Wrap data to signal MessagePack encoding.

    MessagePack is a binary serialization format that's more efficient than JSON.
    Good for: dicts, lists, nested structures.

    Example:
        >>> from kohakuvault import KVault
        >>> from kohakuvault.wrappers import MsgPack
        >>> kv = KVault("data.db")
        >>> kv.enable_auto_pack()
        >>> kv["config"] = MsgPack({"timeout": 30, "retries": 3})
        >>> config = kv["config"]  # → dict (auto-decoded)
    """

    def __init__(self, data):
        super().__init__(data, "msgpack")


class Json(EncodingWrapper):
    """
    Wrap data to signal JSON encoding.

    JSON is human-readable but less efficient than MessagePack.
    Good for: debugging, external tool compatibility.

    Example:
        >>> kv["settings"] = Json({"theme": "dark", "lang": "en"})
    """

    def __init__(self, data):
        super().__init__(data, "json")


class Cbor(EncodingWrapper):
    """
    Wrap data to signal CBOR encoding.

    CBOR is similar to MessagePack but with optional schema validation.

    Example:
        >>> kv["data"] = Cbor({"field1": 123, "field2": "value"})
    """

    def __init__(self, data):
        super().__init__(data, "cbor")


class Pickle(EncodingWrapper):
    """
    Wrap data to explicitly use Pickle encoding.

    Pickle is Python-specific but can serialize arbitrary Python objects.
    Use when: DataPacker doesn't support your type.

    Example:
        >>> kv["custom"] = Pickle(MyCustomClass())
    """

    def __init__(self, data):
        super().__init__(data, "pickle")
