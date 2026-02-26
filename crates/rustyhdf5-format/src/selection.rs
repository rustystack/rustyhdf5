//! Hyperslab and point selection for partial dataset I/O.
//!
//! A [`Selection`] describes which elements of a dataset to read or write.
//! The most common form is a hyperslab — a regular, strided sub-region of
//! the dataspace.
//!
//! # Example
//!
//! ```ignore
//! use rustyhdf5_format::selection::Selection;
//!
//! // Select rows 20..30, columns 40..60 from a 2D dataset
//! let sel = Selection::slice(&[20..30, 40..60]);
//! assert_eq!(sel.num_elements(&[100, 100]), 200); // 10 * 20
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::ops::Range;

/// A selection describing which elements of a dataset to access.
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    /// Select all elements (equivalent to the entire dataspace).
    All,

    /// Select no elements.
    None,

    /// A regular hyperslab selection defined by start, stride, count, and block.
    ///
    /// For each dimension:
    /// - `start[d]` — first element index
    /// - `stride[d]` — step between blocks (must be >= block[d])
    /// - `count[d]` — number of blocks
    /// - `block[d]` — number of consecutive elements per block
    ///
    /// When stride == block (or stride is 1 and block is 1), this reduces
    /// to a simple contiguous slice.
    Hyperslab {
        start: Vec<u64>,
        stride: Vec<u64>,
        count: Vec<u64>,
        block: Vec<u64>,
    },

    /// Select individual points by coordinate.
    Points(Vec<Vec<u64>>),
}

impl Selection {
    /// Create a simple contiguous hyperslab from ranges (one per dimension).
    ///
    /// This is equivalent to a hyperslab with stride=1 and block=1.
    pub fn slice(ranges: &[Range<u64>]) -> Self {
        let rank = ranges.len();
        let mut start = Vec::with_capacity(rank);
        let mut count = Vec::with_capacity(rank);
        for r in ranges {
            start.push(r.start);
            count.push(r.end - r.start);
        }
        Selection::Hyperslab {
            start,
            stride: vec![1; rank],
            count,
            block: vec![1; rank],
        }
    }

    /// Number of selected elements for a given dataspace shape.
    pub fn num_elements(&self, dims: &[u64]) -> u64 {
        match self {
            Selection::All => dims.iter().product(),
            Selection::None => 0,
            Selection::Hyperslab { count, block, .. } => {
                count.iter().zip(block.iter()).map(|(&c, &b)| c * b).product()
            }
            Selection::Points(pts) => pts.len() as u64,
        }
    }

    /// The rank (number of dimensions) of this selection.
    pub fn rank(&self) -> Option<usize> {
        match self {
            Selection::All | Selection::None => Option::None,
            Selection::Hyperslab { start, .. } => Some(start.len()),
            Selection::Points(pts) => pts.first().map(|p| p.len()),
        }
    }

    /// The shape of the selected region (output dimensions).
    ///
    /// For hyperslabs, this is `count[d] * block[d]` per dimension.
    /// For `All`, returns the dataspace shape. For `None`, returns empty.
    pub fn output_shape(&self, dims: &[u64]) -> Vec<u64> {
        match self {
            Selection::All => dims.to_vec(),
            Selection::None => vec![],
            Selection::Hyperslab { count, block, .. } => {
                count.iter().zip(block.iter()).map(|(&c, &b)| c * b).collect()
            }
            Selection::Points(pts) => vec![pts.len() as u64],
        }
    }

    /// Check whether a chunk at the given offset (with given chunk dimensions)
    /// intersects this selection.
    ///
    /// Returns `true` if any element in the chunk overlaps with the selection.
    pub fn intersects_chunk(&self, chunk_offset: &[u64], chunk_dims: &[u64]) -> bool {
        match self {
            Selection::All => true,
            Selection::None => false,
            Selection::Hyperslab {
                start,
                stride,
                count,
                block,
            } => {
                // For each dimension, check if the chunk range overlaps the hyperslab range
                for d in 0..start.len() {
                    let chunk_start = chunk_offset[d];
                    let chunk_end = chunk_start + chunk_dims[d] as u64;

                    // Compute the full extent of the hyperslab in this dimension
                    let sel_start = start[d];
                    let sel_end = if count[d] == 0 {
                        sel_start
                    } else {
                        start[d] + (count[d] - 1) * stride[d] + block[d]
                    };

                    // No overlap if chunk is entirely before or after selection
                    if chunk_end <= sel_start || chunk_start >= sel_end {
                        return false;
                    }
                }
                true
            }
            Selection::Points(pts) => {
                pts.iter().any(|pt| {
                    pt.iter()
                        .zip(chunk_offset.iter().zip(chunk_dims.iter()))
                        .all(|(&p, (&off, &dim))| p >= off && p < off + dim as u64)
                })
            }
        }
    }

    /// For a given chunk, compute the local ranges within the chunk that
    /// overlap with this selection.
    ///
    /// Returns a list of (chunk_local_start, chunk_local_end, output_offset) per
    /// dimension, representing which elements from the chunk contribute to the
    /// output buffer. For simple contiguous slices, this returns exactly one range
    /// per dimension.
    pub fn chunk_local_ranges(
        &self,
        chunk_offset: &[u64],
        chunk_dims: &[u64],
    ) -> Vec<Range<u64>> {
        match self {
            Selection::All => {
                chunk_dims.iter().map(|&d| 0..d as u64).collect()
            }
            Selection::None => vec![],
            Selection::Hyperslab {
                start,
                stride,
                count,
                block,
            } => {
                let mut ranges = Vec::with_capacity(start.len());
                for d in 0..start.len() {
                    let chunk_start = chunk_offset[d];
                    let chunk_end = chunk_start + chunk_dims[d] as u64;

                    // For simple contiguous selections (stride==1, block==1),
                    // just clamp the selection range to the chunk bounds
                    if stride[d] == 1 && block[d] == 1 {
                        let sel_start = start[d];
                        let sel_end = start[d] + count[d];
                        let local_start = sel_start.max(chunk_start) - chunk_start;
                        let local_end = sel_end.min(chunk_end) - chunk_start;
                        ranges.push(local_start..local_end);
                    } else {
                        // General strided case: find all blocks that overlap this chunk
                        let sel_start = start[d];
                        let mut min_local = chunk_dims[d] as u64;
                        let mut max_local = 0u64;

                        for bi in 0..count[d] {
                            let block_start = sel_start + bi * stride[d];
                            let block_end = block_start + block[d];
                            // Check overlap with chunk
                            if block_end > chunk_start && block_start < chunk_end {
                                let local_s = block_start.max(chunk_start) - chunk_start;
                                let local_e = block_end.min(chunk_end) - chunk_start;
                                min_local = min_local.min(local_s);
                                max_local = max_local.max(local_e);
                            }
                        }
                        if max_local > min_local {
                            ranges.push(min_local..max_local);
                        } else {
                            ranges.push(0..0);
                        }
                    }
                }
                ranges
            }
            Selection::Points(_) => {
                // For point selections, return the full chunk range
                // (filtering happens at the element level)
                chunk_dims.iter().map(|&d| 0..d as u64).collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_all_num_elements() {
        let sel = Selection::All;
        assert_eq!(sel.num_elements(&[100, 200]), 20000);
    }

    #[test]
    fn selection_none_num_elements() {
        let sel = Selection::None;
        assert_eq!(sel.num_elements(&[100, 200]), 0);
    }

    #[test]
    fn selection_slice_basic() {
        let sel = Selection::slice(&[20..30, 40..60]);
        assert_eq!(sel.num_elements(&[100, 100]), 200); // 10 * 20
        assert_eq!(sel.output_shape(&[100, 100]), vec![10, 20]);
    }

    #[test]
    fn selection_slice_1d() {
        let sel = Selection::slice(&[5..15]);
        assert_eq!(sel.num_elements(&[100]), 10);
        assert_eq!(sel.output_shape(&[100]), vec![10]);
    }

    #[test]
    fn selection_intersects_chunk_basic() {
        let sel = Selection::slice(&[20..30, 40..60]);

        // Chunk [20..30, 40..50] — overlaps
        assert!(sel.intersects_chunk(&[20, 40], &[10, 10]));

        // Chunk [0..10, 0..10] — no overlap
        assert!(!sel.intersects_chunk(&[0, 0], &[10, 10]));

        // Chunk [20..30, 50..60] — overlaps
        assert!(sel.intersects_chunk(&[20, 50], &[10, 10]));

        // Chunk [30..40, 40..50] — no overlap (just past end in dim 0)
        assert!(!sel.intersects_chunk(&[30, 40], &[10, 10]));
    }

    #[test]
    fn selection_chunk_local_ranges_simple() {
        let sel = Selection::slice(&[25..35, 40..60]);

        // Chunk [20..30, 40..50]
        let ranges = sel.chunk_local_ranges(&[20, 40], &[10, 10]);
        assert_eq!(ranges[0], 5..10); // rows 25..30 within chunk starting at 20
        assert_eq!(ranges[1], 0..10); // cols 40..50 fully selected
    }

    #[test]
    fn selection_points() {
        let sel = Selection::Points(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        assert_eq!(sel.num_elements(&[10, 10]), 3);
        assert_eq!(sel.rank(), Some(2));
    }

    #[test]
    fn selection_all_intersects_any_chunk() {
        let sel = Selection::All;
        assert!(sel.intersects_chunk(&[0, 0], &[10, 10]));
        assert!(sel.intersects_chunk(&[100, 100], &[1, 1]));
    }

    #[test]
    fn selection_hyperslab_strided() {
        // Select every other row: start=0, stride=2, count=5, block=1 in a 10-element dim
        let sel = Selection::Hyperslab {
            start: vec![0],
            stride: vec![2],
            count: vec![5],
            block: vec![1],
        };
        assert_eq!(sel.num_elements(&[10]), 5); // 5 blocks * 1 element each

        // Chunk [0..5] should intersect (contains rows 0, 2, 4)
        assert!(sel.intersects_chunk(&[0], &[5]));
        // Chunk [9..10] should not intersect (only row 9, but selection ends at row 8)
        assert!(!sel.intersects_chunk(&[9], &[1]));
    }
}
