//! Variable-width blocks with u64 offsets and coalesced payload reads.

use rusqlite::{params, Connection, Transaction};

use super::chunks::{self, Block, BLOCK_FIELDS};
use super::core::{new_length, range_count, Meta, MAX_BLOCK_BYTES};
use super::{ColError, ColResult};

pub(crate) fn decode_offsets(bytes: &[u8], payload_len: usize) -> ColResult<Vec<usize>> {
    if bytes.len() < 8 || !bytes.len().is_multiple_of(8) {
        return Err(ColError::Invalid("offsets must contain little-endian u64 values".into()));
    }

    let mut offsets = Vec::with_capacity(bytes.len() / 8);
    for encoded in bytes.chunks_exact(8) {
        let value = usize::try_from(u64::from_le_bytes(encoded.try_into().unwrap()))
            .map_err(|_| ColError::Invalid("offset exceeds address space".into()))?;
        if value > payload_len || offsets.last().is_some_and(|&previous| value < previous) {
            return Err(ColError::Invalid(
                "offsets are not monotonic or exceed the payload".into(),
            ));
        }
        offsets.push(value);
    }
    if offsets[0] != 0 || offsets.last() != Some(&payload_len) {
        return Err(ColError::Invalid("offsets must span the complete payload".into()));
    }
    if offsets
        .windows(2)
        .any(|pair| pair[1] - pair[0] > MAX_BLOCK_BYTES)
    {
        return Err(ColError::Invalid("a variable-width record exceeds 512 MiB".into()));
    }

    Ok(offsets)
}

pub(crate) fn append(
    tx: &Transaction<'_>,
    meta: &Meta,
    payload: &[u8],
    offsets: &[usize],
) -> ColResult<()> {
    if meta.elem_size != 0 {
        return Err(ColError::Invalid("expected a variable-width column".into()));
    }

    let count = offsets.len() - 1;
    let length = new_length(meta.length, count)?;
    let mut tail = chunks::tail(tx, meta.id)?;
    let mut cursor = 0;

    while cursor < count {
        let first_size = offsets[cursor + 1] - offsets[cursor];
        let fits = tail.is_some_and(|block| {
            block.rows < block.row_capacity && first_size <= block.capacity - block.used
        });
        if !fits {
            // Oversized records get a dedicated block; ordinary tails never grow.
            let capacity = meta.block_bytes.max(first_size);
            let row_capacity = if first_size > meta.block_bytes {
                1
            } else {
                (meta.block_bytes / 16).clamp(1, 65_536)
            };
            let number = tail.map_or(0, |block| block.number + 1);
            tail = Some(chunks::allocate(
                tx,
                meta,
                number,
                meta.length + cursor,
                capacity,
                row_capacity,
            )?);
        }

        let mut block = tail.unwrap();
        let begin = cursor;
        while cursor < count && cursor - begin < block.row_capacity - block.rows {
            let needed = offsets[cursor + 1] - offsets[begin];
            if needed > block.capacity - block.used {
                break;
            }
            cursor += 1;
        }
        if cursor == begin {
            return Err(ColError::Corrupt("variable block made no append progress".into()));
        }

        chunks::write_blob(
            tx,
            block.id,
            "payload",
            block.used,
            &payload[offsets[begin]..offsets[cursor]],
        )?;

        let mut encoded = Vec::with_capacity((cursor - begin) * 8);
        for &offset in &offsets[begin + 1..=cursor] {
            encoded
                .extend_from_slice(&((block.used + offset - offsets[begin]) as u64).to_le_bytes());
        }
        chunks::write_blob(tx, block.id, "offsets", (block.rows + 1) * 8, &encoded)?;

        block.rows += cursor - begin;
        block.used += offsets[cursor] - offsets[begin];
        chunks::publish(tx, &block)?;
        tail = Some(block);
    }

    tx.prepare_cached("UPDATE kvault_col_columns SET length = ?2 WHERE col_id = ?1")?
        .execute(params![meta.id, length as i64])?;
    Ok(())
}

pub(crate) fn read(
    conn: &Connection,
    meta: &Meta,
    start: usize,
    count: Option<usize>,
) -> ColResult<(Vec<u8>, Vec<u8>)> {
    if meta.elem_size != 0 {
        return Err(ColError::Invalid("expected a variable-width column".into()));
    }
    let count = range_count(start, count, meta.length)?;
    let mut payload = Vec::new();
    let offset_bytes = count
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| ColError::Invalid("offset buffer size overflow".into()))?;
    let mut output_offsets = Vec::with_capacity(offset_bytes);
    output_offsets.extend_from_slice(&0u64.to_le_bytes());
    if count == 0 {
        return Ok((payload, output_offsets));
    }

    // Use the row index to locate the first block without scanning earlier blocks.
    let first_number: i64 = conn
        .prepare_cached(
            "SELECT block_no FROM kvault_col_blocks
         WHERE col_id = ?1 AND first_row <= ?2 ORDER BY first_row DESC LIMIT 1",
        )?
        .query_row(params![meta.id, start as i64], |row| row.get(0))?;
    let sql = format!(
        "SELECT {BLOCK_FIELDS} FROM kvault_col_blocks
         WHERE col_id = ?1 AND block_no >= ?2 AND first_row < ?3 ORDER BY block_no"
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let blocks = statement
        .query_map(params![meta.id, first_number, (start + count) as i64], Block::from_row)?;
    let mut consumed = 0;

    for found in blocks {
        let block = found?.validate()?;
        let block_end = block
            .first_row
            .checked_add(block.rows)
            .ok_or_else(|| ColError::Corrupt("block row range overflow".into()))?;
        let begin = start.max(block.first_row);
        let end = (start + count).min(block_end);
        if end <= begin || begin != start + consumed {
            return Err(ColError::Corrupt("gap in variable-width block directory".into()));
        }

        let rows = end - begin;
        let mut encoded = vec![0; (rows + 1) * 8];
        chunks::read_blob(conn, block.id, "offsets", (begin - block.first_row) * 8, &mut encoded)?;
        let positions: Vec<usize> = encoded
            .chunks_exact(8)
            .map(|part| u64::from_le_bytes(part.try_into().unwrap()) as usize)
            .collect();
        if positions.windows(2).any(|pair| pair[0] > pair[1]) || positions[rows] > block.used {
            return Err(ColError::Corrupt("invalid stored variable-width offsets".into()));
        }

        // Fetch one contiguous payload range for all requested rows in this block.
        let base = payload.len();
        let size = positions[rows] - positions[0];
        let end_bytes = base
            .checked_add(size)
            .ok_or_else(|| ColError::Invalid("payload size overflow".into()))?;
        payload.resize(end_bytes, 0);
        chunks::read_blob(conn, block.id, "payload", positions[0], &mut payload[base..])?;
        for &offset in &positions[1..] {
            output_offsets
                .extend_from_slice(&((base + offset - positions[0]) as u64).to_le_bytes());
        }
        consumed += rows;
    }

    if consumed != count {
        return Err(ColError::Corrupt("incomplete variable-width read".into()));
    }
    Ok((payload, output_offsets))
}
