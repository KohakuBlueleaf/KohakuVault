//! Shared metadata table for tracking versions and features
//!
//! The kohakuvault_meta table is shared by all components (KVault, ColumnVault, VectorKVault)
//! to track schema versions and supported features.

use rusqlite::{Connection, OptionalExtension};

/// Metadata keys used by different components
#[allow(dead_code)] // Used by ColumnVault
pub const SCHEMA_VERSION_KEY: &str = "schema_version";
pub const KV_FEATURES_KEY: &str = "kv_features";
#[allow(dead_code)] // Used by ColumnVault
pub const COL_FEATURES_KEY: &str = "col_features";

/// Feature flags for KVault
pub const KV_FEATURE_HEADERS: &str = "headers_v1";
#[allow(dead_code)] // Used in Phase 3
pub const KV_FEATURE_AUTO_PACK: &str = "auto_pack_v1";

/// Metadata table manager
pub struct MetaTable;

impl MetaTable {
    /// Create metadata table if it doesn't exist
    pub fn ensure_table(conn: &Connection) -> rusqlite::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS kohakuvault_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
    }

    /// Get a metadata value
    pub fn get(conn: &Connection, key: &str) -> rusqlite::Result<Option<String>> {
        conn.query_row("SELECT value FROM kohakuvault_meta WHERE key = ?", [key], |row| row.get(0))
            .optional()
    }

    /// Set a metadata value (upsert)
    pub fn set(conn: &Connection, key: &str, value: &str) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO kohakuvault_meta (key, value) VALUES (?, ?)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    /// Delete a metadata value
    #[allow(dead_code)] // Utility method for future use
    pub fn delete(conn: &Connection, key: &str) -> rusqlite::Result<()> {
        conn.execute("DELETE FROM kohakuvault_meta WHERE key = ?", [key])?;
        Ok(())
    }

    /// Check if a feature is supported
    ///
    /// Features are stored as comma-separated lists in the meta table
    pub fn has_feature(
        conn: &Connection,
        feature_key: &str,
        feature: &str,
    ) -> rusqlite::Result<bool> {
        if let Some(features_str) = Self::get(conn, feature_key)? {
            Ok(features_str.split(',').any(|f| f == feature))
        } else {
            Ok(false)
        }
    }

    /// Register a feature as supported
    ///
    /// Adds feature to comma-separated list if not already present
    pub fn register_feature(
        conn: &Connection,
        feature_key: &str,
        feature: &str,
    ) -> rusqlite::Result<()> {
        let features_str = if let Some(existing) = Self::get(conn, feature_key)? {
            if existing.split(',').any(|f| f == feature) {
                // Already registered
                return Ok(());
            }
            format!("{},{}", existing, feature)
        } else {
            feature.to_string()
        };

        Self::set(conn, feature_key, &features_str)
    }

    /// Get schema version (returns "0" if not set)
    #[allow(dead_code)] // Used by ColumnVault
    pub fn get_schema_version(conn: &Connection) -> rusqlite::Result<String> {
        Self::get(conn, SCHEMA_VERSION_KEY).map(|v| v.unwrap_or_else(|| "0".to_string()))
    }

    /// Set schema version
    #[allow(dead_code)] // Used by ColumnVault
    pub fn set_schema_version(conn: &Connection, version: &str) -> rusqlite::Result<()> {
        Self::set(conn, SCHEMA_VERSION_KEY, version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_table_basic() {
        let conn = Connection::open_in_memory().unwrap();
        MetaTable::ensure_table(&conn).unwrap();

        // Set and get
        MetaTable::set(&conn, "test_key", "test_value").unwrap();
        let value = MetaTable::get(&conn, "test_key").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        // Get non-existent
        let missing = MetaTable::get(&conn, "missing").unwrap();
        assert_eq!(missing, None);

        // Update existing
        MetaTable::set(&conn, "test_key", "new_value").unwrap();
        let updated = MetaTable::get(&conn, "test_key").unwrap();
        assert_eq!(updated, Some("new_value".to_string()));

        // Delete
        MetaTable::delete(&conn, "test_key").unwrap();
        let deleted = MetaTable::get(&conn, "test_key").unwrap();
        assert_eq!(deleted, None);
    }

    #[test]
    fn test_feature_registration() {
        let conn = Connection::open_in_memory().unwrap();
        MetaTable::ensure_table(&conn).unwrap();

        // Register features
        MetaTable::register_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_HEADERS).unwrap();
        assert!(MetaTable::has_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_HEADERS).unwrap());
        assert!(!MetaTable::has_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_AUTO_PACK).unwrap());

        // Register another feature
        MetaTable::register_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_AUTO_PACK).unwrap();
        assert!(MetaTable::has_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_HEADERS).unwrap());
        assert!(MetaTable::has_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_AUTO_PACK).unwrap());

        // Check features string
        let features = MetaTable::get(&conn, KV_FEATURES_KEY).unwrap().unwrap();
        assert!(features.contains(KV_FEATURE_HEADERS));
        assert!(features.contains(KV_FEATURE_AUTO_PACK));

        // Duplicate registration is idempotent
        MetaTable::register_feature(&conn, KV_FEATURES_KEY, KV_FEATURE_HEADERS).unwrap();
        let features = MetaTable::get(&conn, KV_FEATURES_KEY).unwrap().unwrap();
        assert_eq!(features.matches(KV_FEATURE_HEADERS).count(), 1);
    }

    #[test]
    fn test_schema_version() {
        let conn = Connection::open_in_memory().unwrap();
        MetaTable::ensure_table(&conn).unwrap();

        // Default version is "0"
        let version = MetaTable::get_schema_version(&conn).unwrap();
        assert_eq!(version, "0");

        // Set version
        MetaTable::set_schema_version(&conn, "2").unwrap();
        let version = MetaTable::get_schema_version(&conn).unwrap();
        assert_eq!(version, "2");
    }
}
