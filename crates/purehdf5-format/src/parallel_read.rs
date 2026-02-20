//! Parallel chunk decompression using rayon.
//!
//! When reading a chunked+compressed dataset with many chunks, this module
//! uses rayon's parallel iterators to decompress chunks concurrently.
//! The parallel path is only activated when `chunk_count > 4`.

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::decompress_chunk;

/// Threshold: only use parallel decompression when chunk count exceeds this.
const PARALLEL_THRESHOLD: usize = 4;

/// Result of decompressing a single chunk, tagged with its index for ordering.
struct DecompressedChunk {
    index: usize,
    data: Vec<u8>,
}

/// Returns `true` if the parallel path should be used for the given chunk count.
pub fn should_use_parallel(chunk_count: usize) -> bool {
    chunk_count > PARALLEL_THRESHOLD
}

/// Decompress chunks in parallel using rayon, returning them in order.
///
/// Each chunk is read from `file_data` at the address in the corresponding
/// `ChunkInfo`, decompressed through `pipeline`, and collected in order.
///
/// # Errors
///
/// Returns the first error encountered by any worker thread.
pub fn decompress_chunks_parallel(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: &FilterPipeline,
    chunk_total_bytes: usize,
    element_size: u32,
) -> Result<Vec<Vec<u8>>, FormatError> {
    use rayon::prelude::*;

    let results: Result<Vec<DecompressedChunk>, FormatError> = chunks
        .par_iter()
        .enumerate()
        .map(|(index, chunk_info)| {
            let c_addr = chunk_info.address as usize;
            let size = chunk_info.chunk_size as usize;
            if c_addr + size > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: c_addr + size,
                    available: file_data.len(),
                });
            }
            let raw_chunk = &file_data[c_addr..c_addr + size];

            let decompressed = if chunk_info.filter_mask == 0 {
                decompress_chunk(raw_chunk, pipeline, chunk_total_bytes, element_size)?
            } else {
                raw_chunk.to_vec()
            };

            Ok(DecompressedChunk { index, data: decompressed })
        })
        .collect();

    let mut result_vec = results?;
    result_vec.sort_by_key(|dc| dc.index);
    Ok(result_vec.into_iter().map(|dc| dc.data).collect())
}

/// Decompress chunks sequentially (fallback when parallel is not warranted).
pub fn decompress_chunks_sequential(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    chunk_total_bytes: usize,
    element_size: u32,
) -> Result<Vec<Vec<u8>>, FormatError> {
    let mut result = Vec::with_capacity(chunks.len());
    for chunk_info in chunks {
        let c_addr = chunk_info.address as usize;
        let size = chunk_info.chunk_size as usize;
        if c_addr + size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: c_addr + size,
                available: file_data.len(),
            });
        }
        let raw_chunk = &file_data[c_addr..c_addr + size];

        let decompressed = if let Some(pl) = pipeline {
            if chunk_info.filter_mask == 0 {
                decompress_chunk(raw_chunk, pl, chunk_total_bytes, element_size)?
            } else {
                raw_chunk.to_vec()
            }
        } else {
            raw_chunk.to_vec()
        };
        result.push(decompressed);
    }
    Ok(result)
}
