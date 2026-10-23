# Legacy storage fixture

`legacy_0_9.db.zlib` is a zlib-compressed SQLite database created using the unmodified KohakuVault 0.9.0 extension from commit `20f5fcfc97cc1bea1d3871c829ddba03cea24d19` before the column rewrite.

It contains the `legacy_kv` table (raw bytes, MessagePack mapping, integer, float, string, and a float32 array) and a `legacy_vectors` cosine index with two 4-dimensional vectors and their values. It contains no personal data. Tests decompress it into a temporary directory; they never modify this fixture.

Uncompressed SHA-256: `0afe394adfd4e4a34f6e85fd0f61dc6f46ead7341491b7ad56b47fd6878d4637`.

The fixture is a real old-engine file, not a reconstruction with the new writer.
