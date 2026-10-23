//! Bounded block allocation and incremental BLOB access.

use rusqlite::{params, Connection, DatabaseName, OptionalExtension, Transaction};

use super::core::{Meta, MAX_BLOCK_BYTES};
use super::{ColError, ColResult};

#[derive(Clone, Copy)]
pub(crate) struct Block {
    pub id: i64,
    pub number: i64,
    pub first_row: usize,
    pub rows: usize,
    pub used: usize,
    pub capacity: usize,
    pub row_capacity: usize,
}

impl Block {
    pub(crate) fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            number: row.get(1)?,
            first_row: row.get::<_, i64>(2)? as usize,
            rows: row.get::<_, i64>(3)? as usize,
            used: row.get::<_, i64>(4)? as usize,
            capacity: row.get::<_, i64>(5)? as usize,
            row_capacity: row.get::<_, i64>(6)? as usize,
        })
    }

    pub(crate) fn validate(self) -> ColResult<Self> {
        if self.number < 0
            || self.used > self.capacity
            || self.rows > self.row_capacity
            || self.capacity > MAX_BLOCK_BYTES
            || self.row_capacity > MAX_BLOCK_BYTES
            || self.first_row > i64::MAX as usize
        {
            return Err(ColError::Corrupt("invalid block directory entry".into()));
        }
        Ok(self)
    }
}

pub(crate) const BLOCK_FIELDS: &str =
    "block_id, block_no, first_row, row_count, used_bytes, capacity_bytes, row_capacity";

pub(crate) fn find(conn: &Connection, col_id: i64, number: i64) -> ColResult<Option<Block>> {
    let sql =
        format!("SELECT {BLOCK_FIELDS} FROM kvault_col_blocks WHERE col_id = ?1 AND block_no = ?2");
    let mut stmt = conn.prepare_cached(&sql)?;
    let block = stmt
        .query_row(params![col_id, number], Block::from_row)
        .optional()?;
    block.map(Block::validate).transpose()
}

pub(crate) fn tail(conn: &Connection, col_id: i64) -> ColResult<Option<Block>> {
    let sql = format!("SELECT {BLOCK_FIELDS} FROM kvault_col_blocks WHERE col_id = ?1 ORDER BY block_no DESC LIMIT 1");
    let mut stmt = conn.prepare_cached(&sql)?;
    let block = stmt.query_row([col_id], Block::from_row).optional()?;
    block.map(Block::validate).transpose()
}

pub(crate) fn allocate(
    tx: &Transaction<'_>,
    meta: &Meta,
    number: i64,
    first_row: usize,
    capacity: usize,
    row_capacity: usize,
) -> ColResult<Block> {
    if capacity == 0 || capacity > MAX_BLOCK_BYTES || row_capacity == 0 {
        return Err(ColError::Invalid("invalid block allocation".into()));
    }

    tx.prepare_cached(
        "INSERT INTO kvault_col_blocks
         (col_id, block_no, first_row, row_count, used_bytes, capacity_bytes, row_capacity)
         VALUES (?1, ?2, ?3, 0, 0, ?4, ?5)",
    )?
    .execute(params![
        meta.id,
        number,
        first_row as i64,
        capacity as i64,
        row_capacity as i64
    ])?;
    let id = tx.last_insert_rowid();

    // Keep mutable directory fields out of the row containing large BLOBs.
    if meta.elem_size == 0 {
        let offset_bytes = (row_capacity + 1)
            .checked_mul(8)
            .filter(|&bytes| bytes <= MAX_BLOCK_BYTES)
            .ok_or_else(|| ColError::Invalid("offset buffer too large".into()))?;
        tx.prepare_cached("INSERT INTO kvault_col_data VALUES (?1, zeroblob(?2), zeroblob(?3))")?
            .execute(params![id, capacity as i64, offset_bytes as i64])?;
    } else {
        tx.prepare_cached("INSERT INTO kvault_col_data VALUES (?1, zeroblob(?2), NULL)")?
            .execute(params![id, capacity as i64])?;
    }

    Ok(Block { id, number, first_row, rows: 0, used: 0, capacity, row_capacity })
}

pub(crate) fn publish(tx: &Transaction<'_>, block: &Block) -> ColResult<()> {
    tx.prepare_cached(
        "UPDATE kvault_col_blocks SET row_count = ?2, used_bytes = ?3 WHERE block_id = ?1",
    )?
    .execute(params![block.id, block.rows as i64, block.used as i64])?;
    Ok(())
}

pub(crate) fn read_blob(
    conn: &Connection,
    block_id: i64,
    column: &str,
    offset: usize,
    output: &mut [u8],
) -> ColResult<()> {
    if output.is_empty() {
        return Ok(());
    }

    let blob = conn.blob_open(DatabaseName::Main, "kvault_col_data", column, block_id, true)?;
    if offset > blob.len() || output.len() > blob.len() - offset {
        return Err(ColError::Corrupt("BLOB range exceeds block capacity".into()));
    }
    if blob.read_at(output, offset)? != output.len() {
        return Err(ColError::Corrupt("short BLOB read".into()));
    }
    Ok(())
}

pub(crate) fn write_blob(
    tx: &Transaction<'_>,
    block_id: i64,
    column: &str,
    offset: usize,
    input: &[u8],
) -> ColResult<()> {
    if input.is_empty() {
        return Ok(());
    }

    let mut blob = tx.blob_open(DatabaseName::Main, "kvault_col_data", column, block_id, false)?;
    blob.write_at(input, offset)?;
    Ok(())
}
