//! Extensible Array writer: serialize EA layout messages and build EA structures.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
use crate::chunked_write::WrittenChunk;

/// Serialize a v4 Extensible Array layout message.
pub(crate) fn serialize_v4_extensible_array(
    chunk_dims: &[u32],
    ea_address: u64,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked
    buf.push(0x00); // flags

    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    for &d in chunk_dims {
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 4 (Extensible Array)
    buf.push(4);

    // EA creation parameters (must match AEHD and HDF5 C library defaults)
    buf.push(32); // max_nelmts_bits
    buf.push(4); // idx_blk_elmts
    buf.push(4); // super_blk_min_data_ptrs
    buf.push(16); // data_blk_min_elmts
    buf.push(10); // max_dblk_page_nelmts_bits

    // EA header address
    match offset_size {
        4 => buf.extend_from_slice(&(ea_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&ea_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Build a complete Extensible Array at a known absolute address.
///
/// For simplicity, we put all elements inline in the index block when the
/// number of chunks is small (up to idx_blk_elmts), otherwise use inline +
/// direct data blocks.
pub fn build_extensible_array_at(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    ea_base_address: u64,
) -> Vec<u8> {
    let os = offset_size as usize;
    let num_elements = chunks.len();

    // Compute element encoding size (same logic as Fixed Array)
    let chunk_size_bytes: usize = if has_filters {
        let max_raw = chunks.iter().map(|c| c.raw_size).max().unwrap_or(1);
        let log2_val = if max_raw <= 1 {
            0
        } else {
            63 - max_raw.leading_zeros()
        };
        let len = 1 + ((log2_val + 8) / 8) as usize;
        len.min(8)
    } else {
        0
    };

    let elem_size = if has_filters {
        os + chunk_size_bytes + 4
    } else {
        os
    };

    let client_id: u8 = if has_filters { 1 } else { 0 };

    // EA creation parameters — must match HDF5 C library defaults exactly
    let max_nelmts_bits: u8 = 32;
    let idx_blk_elmts: u8 = 4;
    let min_dblk_nelmts: u8 = 16;
    let super_blk_min_nelmts: u8 = 4;
    let max_dblk_nelmts_bits: u8 = 10;

    // EAHD size: fixed(12) + 6 stats(6*length_size) + addr(offset_size) + checksum(4)
    let aehd_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        + 6 * length_size as usize + os + 4;
    let aeib_address = ea_base_address + aehd_size as u64;

    // Determine how many elements go inline vs data blocks
    let n_inline = (idx_blk_elmts as usize).min(num_elements);
    let remaining_after_inline = num_elements.saturating_sub(n_inline);

    // Compute super block layout per HDF5 spec
    let sblk_min = super_blk_min_nelmts as usize;
    let log2_dblk_min = if min_dblk_nelmts <= 1 { 0 } else { (min_dblk_nelmts as u32).trailing_zeros() as usize };
    let nsblks = (max_nelmts_bits as usize).saturating_sub(log2_dblk_min) + 1;

    // Direct data block addresses (from super blocks 0..sblk_min-1)
    let mut dblk_sizes: Vec<usize> = Vec::new();
    for sblk_idx in 0..sblk_min.min(nsblks) {
        let ndblks = 1usize << (sblk_idx / 2);
        let dblk_nelmts = (min_dblk_nelmts as usize) * (1 << sblk_idx.div_ceil(2));
        for _ in 0..ndblks {
            dblk_sizes.push(dblk_nelmts);
        }
    }
    let n_direct_dblks = dblk_sizes.len();

    // Super block addresses (for super blocks sblk_min..nsblks-1)
    let n_sblk_addrs = nsblks.saturating_sub(sblk_min);

    // EAIB size
    let aeib_size = 4 + 1 + 1 + os
        + idx_blk_elmts as usize * elem_size
        + n_direct_dblks * os
        + n_sblk_addrs * os
        + 4;

    // Build AEHD
    let mut aehd = Vec::with_capacity(aehd_size);
    aehd.extend_from_slice(b"EAHD");
    aehd.push(0); // version
    aehd.push(client_id);
    aehd.push(elem_size as u8);
    aehd.push(max_nelmts_bits);
    aehd.push(idx_blk_elmts);
    aehd.push(min_dblk_nelmts);
    aehd.push(super_blk_min_nelmts);
    aehd.push(max_dblk_nelmts_bits);

    // Count data blocks that will have chunks
    let n_active_dblks: u64 = if remaining_after_inline > 0 {
        let mut count = 0u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                count += 1;
                ci += sz;
            }
        }
        count
    } else {
        0
    };
    let aedb_header_overhead = 4 + 1 + 1 + os + 4;
    let data_blk_total_size: u64 = if remaining_after_inline > 0 {
        let mut total = 0u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                total += (aedb_header_overhead + sz * elem_size) as u64;
                ci += sz;
            }
        }
        total
    } else {
        0
    };
    let max_idx_set: u64 = if remaining_after_inline > 0 {
        let mut max_set = idx_blk_elmts as u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                max_set += sz as u64;
                ci += sz;
            }
        }
        max_set
    } else {
        idx_blk_elmts as u64
    };

    let write_length = |buf: &mut Vec<u8>, val: u64| {
        match length_size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&val.to_le_bytes()),
        }
    };
    let write_addr = |buf: &mut Vec<u8>, val: u64| {
        match offset_size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&val.to_le_bytes()),
        }
    };

    write_length(&mut aehd, 0);
    write_length(&mut aehd, 0);
    write_length(&mut aehd, n_active_dblks);
    write_length(&mut aehd, data_blk_total_size);
    write_length(&mut aehd, num_elements as u64);
    write_length(&mut aehd, max_idx_set);

    write_addr(&mut aehd, aeib_address);

    let aehd_checksum = jenkins_lookup3(&aehd);
    aehd.extend_from_slice(&aehd_checksum.to_le_bytes());
    debug_assert_eq!(aehd.len(), aehd_size);

    // Build AEIB
    let mut aeib = Vec::with_capacity(aeib_size);
    aeib.extend_from_slice(b"EAIB");
    aeib.push(0);
    aeib.push(client_id);

    match offset_size {
        4 => aeib.extend_from_slice(&(ea_base_address as u32).to_le_bytes()),
        8 => aeib.extend_from_slice(&ea_base_address.to_le_bytes()),
        _ => aeib.extend_from_slice(&ea_base_address.to_le_bytes()),
    }

    // Inline elements
    #[allow(clippy::needless_range_loop)]
    for i in 0..idx_blk_elmts as usize {
        if i < n_inline {
            write_chunk_element(&mut aeib, &chunks[i], offset_size, has_filters, chunk_size_bytes);
        } else {
            write_undefined_element(&mut aeib, offset_size, has_filters, chunk_size_bytes);
        }
    }

    // Data block addresses + build data blocks
    let mut data_blocks_buf = Vec::new();
    let dblks_base = aeib_address + aeib_size as u64;
    let mut dblk_cursor = dblks_base;
    let mut chunk_idx = n_inline;

    for &nelmts in &dblk_sizes {
        if chunk_idx >= num_elements {
            match offset_size {
                4 => aeib.extend_from_slice(&u32::MAX.to_le_bytes()),
                8 => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
                _ => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
            }
            continue;
        }

        match offset_size {
            4 => aeib.extend_from_slice(&(dblk_cursor as u32).to_le_bytes()),
            8 => aeib.extend_from_slice(&dblk_cursor.to_le_bytes()),
            _ => aeib.extend_from_slice(&dblk_cursor.to_le_bytes()),
        }

        // Build EADB
        let mut aedb = Vec::new();
        aedb.extend_from_slice(b"EADB");
        aedb.push(0);
        aedb.push(client_id);
        match offset_size {
            4 => aedb.extend_from_slice(&(ea_base_address as u32).to_le_bytes()),
            8 => aedb.extend_from_slice(&ea_base_address.to_le_bytes()),
            _ => aedb.extend_from_slice(&ea_base_address.to_le_bytes()),
        }

        let blk_off_size = (max_nelmts_bits as usize).div_ceil(8);
        let blk_off_val = (chunk_idx - n_inline) as u64;
        aedb.extend_from_slice(&blk_off_val.to_le_bytes()[..blk_off_size]);

        for slot in 0..nelmts {
            if chunk_idx + slot < num_elements {
                write_chunk_element(
                    &mut aedb,
                    &chunks[chunk_idx + slot],
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                );
            } else {
                write_undefined_element(&mut aedb, offset_size, has_filters, chunk_size_bytes);
            }
        }

        let aedb_checksum = jenkins_lookup3(&aedb);
        aedb.extend_from_slice(&aedb_checksum.to_le_bytes());

        dblk_cursor += aedb.len() as u64;
        data_blocks_buf.extend_from_slice(&aedb);
        chunk_idx += nelmts;
    }

    // Super block addresses (all undefined)
    for _ in 0..n_sblk_addrs {
        match offset_size {
            4 => aeib.extend_from_slice(&u32::MAX.to_le_bytes()),
            8 => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
            _ => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
        }
    }

    let aeib_checksum = jenkins_lookup3(&aeib);
    aeib.extend_from_slice(&aeib_checksum.to_le_bytes());
    debug_assert_eq!(aeib.len(), aeib_size);

    let mut combined = aehd;
    combined.extend_from_slice(&aeib);
    combined.extend_from_slice(&data_blocks_buf);
    combined
}

fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &WrittenChunk,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    match offset_size {
        4 => buf.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk.address.to_le_bytes()),
        _ => buf.extend_from_slice(&chunk.address.to_le_bytes()),
    }
    if has_filters {
        let cs_bytes = chunk.compressed_size.to_le_bytes();
        buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
        buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
    }
}

fn write_undefined_element(
    buf: &mut Vec<u8>,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    let os = offset_size as usize;
    buf.extend_from_slice(&vec![0xFF; os]);
    if has_filters {
        buf.extend_from_slice(&vec![0x00; chunk_size_bytes]);
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}
