//! Chunk B-tree index cache and pre-computed layout for fast repeated reads.
//!
//! [`ChunkIndex`] scans the chunk index structure (B-tree v1, fixed array, etc.)
//! once and builds a flat `HashMap<ChunkCoord, ChunkInfo>` for O(1) lookups.
//!
//! [`ChunkLayout`] pre-computes contiguous row-copy operations for each chunk,
//! turning the N-D → flat coordinate math into a series of `memcpy` calls.
//! This eliminates per-element coordinate arithmetic during output assembly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;

use crate::chunk_cache::ChunkCoord;
use crate::chunked_read::ChunkInfo;

// ---------------------------------------------------------------------------
// ChunkIndex — flat HashMap of coord → ChunkInfo
// ---------------------------------------------------------------------------

/// A cached chunk index that maps chunk coordinates to file location metadata.
///
/// Built once from a B-tree (or other index structure) traversal, then provides
/// O(1) lookups for any chunk coordinate.
pub struct ChunkIndex {
    #[cfg(feature = "std")]
    map: HashMap<ChunkCoord, ChunkInfo>,
    #[cfg(not(feature = "std"))]
    map: BTreeMap<ChunkCoord, ChunkInfo>,
    rank: usize,
}

impl ChunkIndex {
    /// Build a chunk index from a pre-collected list of `ChunkInfo`.
    ///
    /// `rank` is the spatial dimensionality — offsets are truncated to the first
    /// `rank` elements (B-tree v1 stores rank+1 offsets with element size as the
    /// last dimension).
    pub fn build(chunks: &[ChunkInfo], rank: usize) -> Self {
        #[cfg(feature = "std")]
        let mut map = HashMap::with_capacity(chunks.len());
        #[cfg(not(feature = "std"))]
        let mut map = BTreeMap::new();

        for ci in chunks {
            let coord: ChunkCoord = ci.offsets.iter().take(rank).copied().collect();
            map.insert(coord, ci.clone());
        }
        Self { map, rank }
    }

    /// O(1) lookup of a chunk by its spatial coordinate.
    #[inline]
    pub fn lookup(&self, coords: &[u64]) -> Option<&ChunkInfo> {
        self.map.get(coords)
    }

    /// Number of chunks in the index.
    #[inline]
    pub fn num_chunks(&self) -> usize {
        self.map.len()
    }

    /// The spatial rank this index was built for.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Iterate over all indexed chunks.
    pub fn iter(&self) -> impl Iterator<Item = (&ChunkCoord, &ChunkInfo)> {
        self.map.iter()
    }

    /// Return all chunk infos as a Vec (order unspecified).
    pub fn all_chunks(&self) -> Vec<ChunkInfo> {
        self.map.values().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// ChunkLayout — pre-computed row-copy operations
// ---------------------------------------------------------------------------

/// A single contiguous memory copy within the output assembly.
#[derive(Debug, Clone)]
pub struct RowCopy {
    /// Byte offset within the decompressed chunk buffer.
    pub src_offset: usize,
    /// Byte offset within the output buffer.
    pub dst_offset: usize,
    /// Number of bytes to copy.
    pub len: usize,
}

/// Pre-computed mapping for one chunk: its metadata plus the row-copy plan.
#[derive(Debug, Clone)]
pub struct ChunkMapping {
    /// Spatial coordinate of this chunk.
    pub coord: ChunkCoord,
    /// File byte address of the compressed chunk.
    pub file_offset: u64,
    /// Size of the compressed chunk in the file.
    pub file_size: u32,
    /// Filter mask (0 = all filters applied).
    pub filter_mask: u32,
    /// Pre-computed row-copy operations for assembling this chunk into output.
    pub copies: Vec<RowCopy>,
}

/// Pre-computed layout for assembling all chunks of a dataset into a flat
/// output buffer.
///
/// Built once from the chunk index + dataset/chunk dimensions, then reused
/// on every read. Turns the per-element N-D → flat coordinate math into a
/// sequence of `memcpy` calls.
pub struct ChunkLayout {
    pub mappings: Vec<ChunkMapping>,
    /// Total bytes of the output buffer.
    pub output_bytes: usize,
    /// Element size in bytes.
    pub elem_size: usize,
    /// Chunk total bytes (uncompressed).
    pub chunk_total_bytes: usize,
}

impl ChunkLayout {
    /// Build the chunk layout from a chunk index and dataset geometry.
    ///
    /// - `index`: the chunk index (coord → ChunkInfo)
    /// - `ds_dims`: dataset dimensions (e.g. [100, 200])
    /// - `chunk_dims`: chunk dimensions (e.g. [10, 20])
    /// - `elem_size`: size of a single element in bytes
    pub fn build(
        index: &ChunkIndex,
        ds_dims: &[usize],
        chunk_dims: &[usize],
        elem_size: usize,
    ) -> Self {
        let rank = ds_dims.len();

        // Compute dataset strides (row-major)
        let mut ds_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
        }

        // Compute chunk strides (row-major within a chunk)
        let mut chunk_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
        }

        let chunk_total_elements: usize = chunk_dims.iter().product();
        let chunk_total_bytes = chunk_total_elements * elem_size;
        let total_elements: usize = ds_dims.iter().product();
        let output_bytes = total_elements * elem_size;

        let mut mappings = Vec::with_capacity(index.num_chunks());

        for (_coord, ci) in index.iter() {
            let coord: ChunkCoord = ci.offsets.iter().take(rank).copied().collect();
            let chunk_offsets: Vec<usize> = coord.iter().map(|&o| o as usize).collect();

            let copies = if rank == 0 {
                // Scalar dataset — single copy
                let len = chunk_total_bytes.min(output_bytes);
                vec![RowCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    len,
                }]
            } else {
                compute_row_copies(
                    &chunk_offsets,
                    chunk_dims,
                    ds_dims,
                    &ds_strides,
                    &chunk_strides,
                    elem_size,
                    rank,
                )
            };

            mappings.push(ChunkMapping {
                coord,
                file_offset: ci.address,
                file_size: ci.chunk_size,
                filter_mask: ci.filter_mask,
                copies,
            });
        }

        Self {
            mappings,
            output_bytes,
            elem_size,
            chunk_total_bytes,
        }
    }

    /// Assemble output from pre-decompressed chunk buffers using the
    /// pre-computed row-copy plan.
    ///
    /// `chunk_data` must be in the same order as `self.mappings`.
    pub fn assemble(&self, chunk_data: &[&[u8]], output: &mut [u8]) {
        debug_assert_eq!(chunk_data.len(), self.mappings.len());
        for (mapping, data) in self.mappings.iter().zip(chunk_data.iter()) {
            for copy in &mapping.copies {
                let src_end = copy.src_offset + copy.len;
                let dst_end = copy.dst_offset + copy.len;
                if src_end <= data.len() && dst_end <= output.len() {
                    output[copy.dst_offset..dst_end]
                        .copy_from_slice(&data[copy.src_offset..src_end]);
                }
            }
        }
    }
}

/// Compute contiguous row-copy operations for a single chunk.
///
/// The innermost dimension is always contiguous in both chunk and output
/// memory (row-major layout). We iterate over the outer dimensions to find
/// the valid range of the innermost dim, then emit one `RowCopy` per
/// contiguous run.
fn compute_row_copies(
    chunk_offsets: &[usize],
    chunk_dims: &[usize],
    ds_dims: &[usize],
    ds_strides: &[usize],
    chunk_strides: &[usize],
    elem_size: usize,
    rank: usize,
) -> Vec<RowCopy> {
    // Number of "rows" = product of all dimensions except the innermost
    let inner_dim = rank - 1;
    let inner_chunk_len = chunk_dims[inner_dim];
    let inner_ds_len = ds_dims[inner_dim];
    let inner_chunk_offset = chunk_offsets[inner_dim];

    // How many elements of the innermost dim are valid (not out-of-bounds)?
    let inner_valid = if inner_chunk_offset >= inner_ds_len {
        0
    } else {
        inner_chunk_len.min(inner_ds_len - inner_chunk_offset)
    };

    if inner_valid == 0 {
        return Vec::new();
    }

    // Number of outer "rows" (product of dims 0..rank-1)
    let outer_count: usize = chunk_dims[..inner_dim].iter().product();
    let mut copies = Vec::with_capacity(outer_count);

    for outer_flat in 0..outer_count {
        // Convert outer flat index to N-D coordinates in dims 0..rank-1
        let mut remaining = outer_flat;
        let mut ds_flat_base = 0usize;
        let mut chunk_flat_base = 0usize;
        let mut out_of_bounds = false;

        for d in 0..inner_dim {
            let coord_in_chunk = remaining / chunk_strides[d];
            // Only divide by inner strides for dims before inner_dim
            // chunk_strides already accounts for this
            remaining %= chunk_strides[d];

            let global_coord = chunk_offsets[d] + coord_in_chunk;
            if global_coord >= ds_dims[d] {
                out_of_bounds = true;
                break;
            }
            ds_flat_base += global_coord * ds_strides[d];
            chunk_flat_base += coord_in_chunk * chunk_strides[d];
        }

        if out_of_bounds {
            continue;
        }

        // Add the innermost dimension contribution
        ds_flat_base += inner_chunk_offset * ds_strides[inner_dim];
        // chunk_flat_base already aligned to start of inner row

        let src_offset = chunk_flat_base * elem_size;
        let dst_offset = ds_flat_base * elem_size;
        let len = inner_valid * elem_size;

        copies.push(RowCopy {
            src_offset,
            dst_offset,
            len,
        });
    }

    copies
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_read::ChunkInfo;

    fn make_chunk(offsets: Vec<u64>, address: u64, size: u32) -> ChunkInfo {
        ChunkInfo {
            chunk_size: size,
            filter_mask: 0,
            offsets,
            address,
        }
    }

    // --- ChunkIndex tests ---

    #[test]
    fn index_build_and_lookup() {
        let chunks = vec![
            make_chunk(vec![0, 0, 4], 0x1000, 80),
            make_chunk(vec![10, 0, 4], 0x2000, 80),
            make_chunk(vec![10, 5, 4], 0x3000, 80),
        ];
        let index = ChunkIndex::build(&chunks, 2);
        assert_eq!(index.num_chunks(), 3);
        assert_eq!(index.rank(), 2);

        let c0 = index.lookup(&[0, 0]).unwrap();
        assert_eq!(c0.address, 0x1000);

        let c1 = index.lookup(&[10, 0]).unwrap();
        assert_eq!(c1.address, 0x2000);

        let c2 = index.lookup(&[10, 5]).unwrap();
        assert_eq!(c2.address, 0x3000);

        assert!(index.lookup(&[5, 0]).is_none());
    }

    #[test]
    fn index_empty() {
        let index = ChunkIndex::build(&[], 2);
        assert_eq!(index.num_chunks(), 0);
        assert!(index.lookup(&[0, 0]).is_none());
    }

    #[test]
    fn index_1d() {
        let chunks = vec![
            make_chunk(vec![0, 8], 0x100, 40),
            make_chunk(vec![5, 8], 0x200, 40),
        ];
        let index = ChunkIndex::build(&chunks, 1);
        assert_eq!(index.num_chunks(), 2);

        assert_eq!(index.lookup(&[0]).unwrap().address, 0x100);
        assert_eq!(index.lookup(&[5]).unwrap().address, 0x200);
    }

    #[test]
    fn index_3d() {
        let chunks = vec![
            make_chunk(vec![0, 0, 0, 4], 0x100, 100),
            make_chunk(vec![0, 0, 5, 4], 0x200, 100),
            make_chunk(vec![2, 3, 0, 4], 0x300, 100),
        ];
        let index = ChunkIndex::build(&chunks, 3);
        assert_eq!(index.num_chunks(), 3);

        assert_eq!(index.lookup(&[0, 0, 0]).unwrap().address, 0x100);
        assert_eq!(index.lookup(&[0, 0, 5]).unwrap().address, 0x200);
        assert_eq!(index.lookup(&[2, 3, 0]).unwrap().address, 0x300);
    }

    // --- ChunkLayout tests ---

    #[test]
    fn layout_1d_two_chunks() {
        // 20-element 1D dataset, chunk size 10, f64
        let chunks = vec![
            make_chunk(vec![0, 8], 0x1000, 80),
            make_chunk(vec![10, 8], 0x2000, 80),
        ];
        let index = ChunkIndex::build(&chunks, 1);
        let layout = ChunkLayout::build(&index, &[20], &[10], 8);

        assert_eq!(layout.output_bytes, 160); // 20 * 8
        assert_eq!(layout.mappings.len(), 2);

        // Each 1D chunk should have exactly 1 row-copy
        for mapping in &layout.mappings {
            assert_eq!(mapping.copies.len(), 1);
            assert_eq!(mapping.copies[0].len, 80); // 10 * 8
        }

        // Verify assembly
        let chunk0: Vec<u8> = (0..10u64)
            .flat_map(|i| (i as f64).to_le_bytes())
            .collect();
        let chunk1: Vec<u8> = (10..20u64)
            .flat_map(|i| (i as f64).to_le_bytes())
            .collect();

        let mut output = vec![0u8; 160];
        // Find which mapping goes where
        let data_refs: Vec<&[u8]> = layout
            .mappings
            .iter()
            .map(|m| {
                if m.coord == vec![0] {
                    chunk0.as_slice()
                } else {
                    chunk1.as_slice()
                }
            })
            .collect();
        layout.assemble(&data_refs, &mut output);

        for i in 0..20u64 {
            let val = f64::from_le_bytes(
                output[i as usize * 8..(i as usize + 1) * 8]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    #[test]
    fn layout_2d_four_chunks() {
        // 4x6 dataset, chunk 2x3, f32 (4 bytes)
        let chunks = vec![
            make_chunk(vec![0, 0, 4], 0x100, 24),
            make_chunk(vec![0, 3, 4], 0x200, 24),
            make_chunk(vec![2, 0, 4], 0x300, 24),
            make_chunk(vec![2, 3, 4], 0x400, 24),
        ];
        let index = ChunkIndex::build(&chunks, 2);
        let layout = ChunkLayout::build(&index, &[4, 6], &[2, 3], 4);

        assert_eq!(layout.output_bytes, 96); // 24 * 4
        assert_eq!(layout.mappings.len(), 4);

        // Each 2x3 chunk should have 2 row-copies (one per row)
        for mapping in &layout.mappings {
            assert_eq!(mapping.copies.len(), 2);
            // Each copy should be 3 elements * 4 bytes = 12 bytes
            for copy in &mapping.copies {
                assert_eq!(copy.len, 12);
            }
        }

        // Build chunk data and verify assembly
        let mut chunk_data_map: Vec<(Vec<u64>, Vec<u8>)> = Vec::new();
        for row_start in [0usize, 2] {
            for col_start in [0usize, 3] {
                let mut data = Vec::new();
                for r in 0..2 {
                    for c in 0..3 {
                        let val = ((row_start + r) * 6 + (col_start + c)) as f32;
                        data.extend_from_slice(&val.to_le_bytes());
                    }
                }
                chunk_data_map.push((vec![row_start as u64, col_start as u64], data));
            }
        }

        let data_refs: Vec<&[u8]> = layout
            .mappings
            .iter()
            .map(|m| {
                chunk_data_map
                    .iter()
                    .find(|(c, _)| c == &m.coord)
                    .unwrap()
                    .1
                    .as_slice()
            })
            .collect();

        let mut output = vec![0u8; 96];
        layout.assemble(&data_refs, &mut output);

        for i in 0..24 {
            let val = f32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(val, i as f32, "mismatch at element {i}");
        }
    }

    #[test]
    fn layout_partial_last_chunk() {
        // 25-element 1D, chunk 10 => 3 chunks, last has 5 valid
        let chunks = vec![
            make_chunk(vec![0, 8], 0x100, 80),
            make_chunk(vec![10, 8], 0x200, 80),
            make_chunk(vec![20, 8], 0x300, 80),
        ];
        let index = ChunkIndex::build(&chunks, 1);
        let layout = ChunkLayout::build(&index, &[25], &[10], 8);

        assert_eq!(layout.output_bytes, 200); // 25 * 8

        // Find the chunk at offset 20 — should only copy 5 elements
        let partial = layout
            .mappings
            .iter()
            .find(|m| m.coord == vec![20])
            .unwrap();
        assert_eq!(partial.copies.len(), 1);
        assert_eq!(partial.copies[0].len, 40); // 5 * 8 bytes
    }

    #[test]
    fn layout_2d_partial_boundary() {
        // 5x5 dataset, chunk 3x3 => 4 chunks, some partial
        let chunks = vec![
            make_chunk(vec![0, 0, 4], 0x100, 36),
            make_chunk(vec![0, 3, 4], 0x200, 36),
            make_chunk(vec![3, 0, 4], 0x300, 36),
            make_chunk(vec![3, 3, 4], 0x400, 36),
        ];
        let index = ChunkIndex::build(&chunks, 2);
        let layout = ChunkLayout::build(&index, &[5, 5], &[3, 3], 4);

        assert_eq!(layout.output_bytes, 100); // 25 * 4

        // Chunk at (0,0): 3x3 fully valid → 3 rows, 3 elems each
        let c00 = layout
            .mappings
            .iter()
            .find(|m| m.coord == vec![0, 0])
            .unwrap();
        assert_eq!(c00.copies.len(), 3);
        for copy in &c00.copies {
            assert_eq!(copy.len, 12); // 3 * 4
        }

        // Chunk at (0,3): 3x2 valid (cols 3,4) → 3 rows, 2 elems each
        let c03 = layout
            .mappings
            .iter()
            .find(|m| m.coord == vec![0, 3])
            .unwrap();
        assert_eq!(c03.copies.len(), 3);
        for copy in &c03.copies {
            assert_eq!(copy.len, 8); // 2 * 4
        }

        // Chunk at (3,0): 2x3 valid (rows 3,4) → 2 rows, 3 elems each
        let c30 = layout
            .mappings
            .iter()
            .find(|m| m.coord == vec![3, 0])
            .unwrap();
        assert_eq!(c30.copies.len(), 2);

        // Chunk at (3,3): 2x2 valid → 2 rows, 2 elems each
        let c33 = layout
            .mappings
            .iter()
            .find(|m| m.coord == vec![3, 3])
            .unwrap();
        assert_eq!(c33.copies.len(), 2);
        for copy in &c33.copies {
            assert_eq!(copy.len, 8); // 2 * 4
        }
    }

    #[test]
    fn layout_3d_basic() {
        // 4x4x4 dataset, chunk 2x2x2, f32 (4 bytes) => 8 chunks
        let mut chunks = Vec::new();
        let elem_size = 4u32;
        let chunk_bytes = 2 * 2 * 2 * elem_size;
        let mut addr = 0x1000u64;
        for x in [0u64, 2] {
            for y in [0u64, 2] {
                for z in [0u64, 2] {
                    chunks.push(make_chunk(
                        vec![x, y, z, elem_size as u64],
                        addr,
                        chunk_bytes,
                    ));
                    addr += chunk_bytes as u64;
                }
            }
        }

        let index = ChunkIndex::build(&chunks, 3);
        let layout = ChunkLayout::build(&index, &[4, 4, 4], &[2, 2, 2], 4);

        assert_eq!(layout.output_bytes, 256); // 64 * 4
        assert_eq!(layout.mappings.len(), 8);

        // Each 2x2x2 chunk: outer dims are 2x2=4 rows, each row copies 2 elements
        for mapping in &layout.mappings {
            assert_eq!(mapping.copies.len(), 4); // 2*2 outer rows
            for copy in &mapping.copies {
                assert_eq!(copy.len, 8); // 2 * 4 bytes
            }
        }
    }

    #[test]
    fn assemble_matches_element_by_element() {
        // Verify that layout assembly produces identical output to the
        // original copy_chunk_to_output approach (2D case)
        let ds_dims = [6usize, 8];
        let chunk_dims = [3usize, 4];
        let elem_size = 8usize; // f64

        let mut chunks = Vec::new();
        let mut addr = 0x1000u64;
        for r in (0..6).step_by(3) {
            for c in (0..8).step_by(4) {
                chunks.push(make_chunk(
                    vec![r as u64, c as u64, elem_size as u64],
                    addr,
                    (3 * 4 * elem_size) as u32,
                ));
                addr += (3 * 4 * elem_size) as u64;
            }
        }

        let index = ChunkIndex::build(&chunks, 2);
        let layout = ChunkLayout::build(&index, &ds_dims, &chunk_dims, elem_size);

        // Build chunk data
        let mut chunk_buffers: Vec<(Vec<u64>, Vec<u8>)> = Vec::new();
        for mapping in &layout.mappings {
            let row_start = mapping.coord[0] as usize;
            let col_start = mapping.coord[1] as usize;
            let mut buf = vec![0u8; 3 * 4 * elem_size];
            for r in 0..3 {
                for c in 0..4 {
                    let val = ((row_start + r) * 8 + (col_start + c)) as f64;
                    let offset = (r * 4 + c) * elem_size;
                    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
                }
            }
            chunk_buffers.push((mapping.coord.clone(), buf));
        }

        let data_refs: Vec<&[u8]> = chunk_buffers.iter().map(|(_, b)| b.as_slice()).collect();
        let mut output = vec![0u8; 6 * 8 * elem_size];
        layout.assemble(&data_refs, &mut output);

        // Verify every element
        for r in 0..6 {
            for c in 0..8 {
                let idx = r * 8 + c;
                let val = f64::from_le_bytes(
                    output[idx * 8..(idx + 1) * 8].try_into().unwrap(),
                );
                assert_eq!(val, idx as f64, "mismatch at ({r}, {c})");
            }
        }
    }
}
