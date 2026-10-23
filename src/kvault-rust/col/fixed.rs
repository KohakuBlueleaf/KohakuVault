//! Contiguous fixed-width append, range reads, and indexed gathers.

use rusqlite::{params, Connection, DatabaseName, Transaction};

use super::chunks;
use super::core::{new_length, range_count, Meta};
use super::{ColError, ColResult};

fn rows_per_block(meta: &Meta) -> ColResult<usize> {
    if meta.elem_size == 0 {
        return Err(ColError::Invalid("expected a fixed-width column".into()));
    }
    Ok((meta.block_bytes / meta.elem_size).max(1))
}

pub(crate) fn append(tx: &Transaction<'_>, meta: &Meta, input: &[u8]) -> ColResult<()> {
    let block_rows = rows_per_block(meta)?;
    if !input.len().is_multiple_of(meta.elem_size) {
        return Err(ColError::Invalid("buffer is not aligned to the column element size".into()));
    }

    let count = input.len() / meta.elem_size;
    let length = new_length(meta.length, count)?;
    let mut consumed = 0;

    while consumed < count {
        let position = meta.length + consumed;
        let number = position / block_rows;
        let local = position % block_rows;
        let take = (count - consumed).min(block_rows - local);
        let mut block = match chunks::find(tx, meta.id, number as i64)? {
            Some(block) => block,
            None => chunks::allocate(
                tx,
                meta,
                number as i64,
                number * block_rows,
                block_rows * meta.elem_size,
                block_rows,
            )?,
        };
        if block.rows != local {
            return Err(ColError::Corrupt("append position disagrees with block length".into()));
        }

        // Write each contiguous part once, then publish only small metadata.
        chunks::write_blob(
            tx,
            block.id,
            "payload",
            local * meta.elem_size,
            &input[consumed * meta.elem_size..(consumed + take) * meta.elem_size],
        )?;
        block.rows += take;
        block.used = block.rows * meta.elem_size;
        chunks::publish(tx, &block)?;
        consumed += take;
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
) -> ColResult<Vec<u8>> {
    let block_rows = rows_per_block(meta)?;
    let count = range_count(start, count, meta.length)?;
    let bytes = count
        .checked_mul(meta.elem_size)
        .ok_or_else(|| ColError::Invalid("read size overflow".into()))?;
    let mut output = vec![0; bytes];
    let mut consumed = 0;

    while consumed < count {
        let position = start + consumed;
        let local = position % block_rows;
        let take = (count - consumed).min(block_rows - local);
        let block = chunks::find(conn, meta.id, (position / block_rows) as i64)?
            .ok_or_else(|| ColError::Corrupt("missing fixed-width block".into()))?;
        if local + take > block.rows {
            return Err(ColError::Corrupt("read exceeds committed block rows".into()));
        }

        chunks::read_blob(
            conn,
            block.id,
            "payload",
            local * meta.elem_size,
            &mut output[consumed * meta.elem_size..(consumed + take) * meta.elem_size],
        )?;
        consumed += take;
    }

    Ok(output)
}

pub(crate) fn overwrite(
    tx: &Transaction<'_>,
    meta: &Meta,
    start: usize,
    input: &[u8],
) -> ColResult<()> {
    let block_rows = rows_per_block(meta)?;
    if !input.len().is_multiple_of(meta.elem_size) {
        return Err(ColError::Invalid("replacement is not element aligned".into()));
    }
    let count = input.len() / meta.elem_size;
    range_count(start, Some(count), meta.length)?;
    let mut consumed = 0;

    while consumed < count {
        let position = start + consumed;
        let local = position % block_rows;
        let take = (count - consumed).min(block_rows - local);
        let block = chunks::find(tx, meta.id, (position / block_rows) as i64)?
            .ok_or_else(|| ColError::Corrupt("missing fixed-width block".into()))?;

        chunks::write_blob(
            tx,
            block.id,
            "payload",
            local * meta.elem_size,
            &input[consumed * meta.elem_size..(consumed + take) * meta.elem_size],
        )?;
        consumed += take;
    }

    Ok(())
}

pub(crate) fn take(conn: &Connection, meta: &Meta, indices: &[i64]) -> ColResult<Vec<u8>> {
    let block_rows = rows_per_block(meta)?;
    let bytes = indices
        .len()
        .checked_mul(meta.elem_size)
        .ok_or_else(|| ColError::Invalid("gather size overflow".into()))?;
    let mut ordered = Vec::with_capacity(indices.len());
    for (destination, &index) in indices.iter().enumerate() {
        let index = if index < 0 {
            meta.length as i64 + index
        } else {
            index
        };
        if index < 0 || index as usize >= meta.length {
            return Err(ColError::Bounds);
        }
        ordered.push((index as usize, destination));
    }
    ordered.sort_unstable_by_key(|&(index, _)| index);

    // Group random requests by block, retaining caller order and duplicates.
    let mut output = vec![0; bytes];
    let mut cursor = 0;
    while cursor < ordered.len() {
        let number = ordered[cursor].0 / block_rows;
        let block = chunks::find(conn, meta.id, number as i64)?
            .ok_or_else(|| ColError::Corrupt("missing gather block".into()))?;
        let blob =
            conn.blob_open(DatabaseName::Main, "kvault_col_data", "payload", block.id, true)?;

        while cursor < ordered.len() && ordered[cursor].0 / block_rows == number {
            let (index, destination) = ordered[cursor];
            let local = index % block_rows;
            if local >= block.rows {
                return Err(ColError::Corrupt("gather exceeds block length".into()));
            }
            let buffer =
                &mut output[destination * meta.elem_size..(destination + 1) * meta.elem_size];
            if blob.read_at(buffer, local * meta.elem_size)? != buffer.len() {
                return Err(ColError::Corrupt("short gather read".into()));
            }
            cursor += 1;
        }
    }

    Ok(output)
}
