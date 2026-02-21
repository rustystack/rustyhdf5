//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
use crate::ea_writer;
use crate::error::FormatError;
use crate::filter_pipeline::{
    FilterDescription, FilterPipeline, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZ4,
    FILTER_SHUFFLE, FILTER_ZSTD,
};
use crate::filters::compress_chunk;

/// Options for chunked dataset creation.
#[derive(Debug, Clone, Default)]
pub struct ChunkOptions {
    /// Chunk dimensions (one per dataset dimension).
    pub chunk_dims: Option<Vec<u64>>,
    /// Deflate compression level (0-9), None = no deflate.
    pub deflate_level: Option<u32>,
    /// Whether to apply shuffle filter before compression.
    pub shuffle: bool,
    /// Whether to apply fletcher32 checksum.
    pub fletcher32: bool,
    /// Whether to use LZ4 compression (filter ID 32004).
    pub lz4: bool,
    /// Zstandard compression level (1-22), None = no zstd. Filter ID 32015.
    pub zstd_level: Option<u32>,
}

impl ChunkOptions {
    /// Whether any chunking option is enabled.
    pub fn is_chunked(&self) -> bool {
        self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
            || self.lz4
            || self.zstd_level.is_some()
    }

    /// Build a FilterPipeline from the options.
    pub fn build_pipeline(&self, element_size: u32) -> Option<FilterPipeline> {
        let mut filters = Vec::new();

        if self.shuffle {
            filters.push(FilterDescription {
                filter_id: FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![element_size],
            });
        }

        // Compression filters (mutually exclusive, priority: zstd > lz4 > deflate)
        if let Some(level) = self.zstd_level {
            filters.push(FilterDescription {
                filter_id: FILTER_ZSTD,
                name: Some("zstd".into()),
                flags: 0,
                client_data: vec![level],
            });
        } else if self.lz4 {
            filters.push(FilterDescription {
                filter_id: FILTER_LZ4,
                name: Some("lz4".into()),
                flags: 0,
                client_data: vec![],
            });
        } else if let Some(level) = self.deflate_level {
            filters.push(FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![level],
            });
        }

        if self.fletcher32 {
            filters.push(FilterDescription {
                filter_id: FILTER_FLETCHER32,
                name: None,
                flags: 0,
                client_data: vec![],  
            });
        }

        // Note: h5py sets flags=0x0001 (optional) on filters, but this is not required
        // for read compatibility.

        if filters.is_empty() {
            None
        } else {
            Some(FilterPipeline {
                version: 2,
                filters,
            })
        }
    }

    /// Determine chunk dimensions, using user-specified or auto-computing.
    pub fn resolve_chunk_dims(&self, shape: &[u64]) -> Vec<u64> {
        if let Some(ref dims) = self.chunk_dims {
            dims.clone()
        } else {
            // Auto chunk: use the full dataset shape (single chunk)
            shape.to_vec()
        }
    }
}

/// A chunk that has been written to the file buffer.
#[derive(Debug, Clone)]
pub struct WrittenChunk {
    /// Address within the file where chunk data starts.
    pub address: u64,
    /// Size of the (possibly compressed) chunk data in bytes.
    pub compressed_size: u64,
    /// Original uncompressed size in bytes.
    pub raw_size: u64,
    /// Filter mask (0 = all filters applied).
    pub filter_mask: u32,
}

/// Result of building a chunked dataset.
pub struct ChunkedDataResult {
    /// Raw bytes containing all chunk data + index structures.
    pub data_bytes: Vec<u8>,
    /// The DataLayout v4 message bytes.
    pub layout_message: Vec<u8>,
    /// The FilterPipeline message bytes, if any.
    pub pipeline_message: Option<Vec<u8>>,
}

/// Split raw data into chunk-sized pieces based on shape and chunk dimensions.
/// Returns a Vec of (chunk_offset_per_dim, chunk_raw_bytes).
pub fn split_into_chunks(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
) -> Vec<(Vec<u64>, Vec<u8>)> {
    let rank = shape.len();
    if rank == 0 {
        return vec![(vec![], raw_data.to_vec())];
    }

    // Compute number of chunks per dimension
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        num_chunks_per_dim.push(shape[d].div_ceil(chunk_dims[d]));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    // Dataset strides (row-major)
    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * shape[i + 1] as usize;
    }

    // Chunk strides
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1] as usize;
    }

    let chunk_total_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();

    let mut result = Vec::with_capacity(total_chunks as usize);

    for linear_idx in 0..total_chunks {
        // Convert linear index to chunk grid coordinates
        let mut chunk_grid_coords = vec![0u64; rank];
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            chunk_grid_coords[d] = remaining % num_chunks_per_dim[d];
            remaining /= num_chunks_per_dim[d];
        }

        // Chunk offset in dataset space
        let offsets: Vec<u64> = (0..rank)
            .map(|d| chunk_grid_coords[d] * chunk_dims[d])
            .collect();

        // Extract chunk data
        let mut chunk_bytes = vec![0u8; chunk_total_elements * element_size];

        for flat_idx in 0..chunk_total_elements {
            let mut remaining_idx = flat_idx;
            let mut ds_flat = 0usize;
            let mut out_of_bounds = false;

            for d in 0..rank {
                let coord_in_chunk = remaining_idx / chunk_strides[d];
                remaining_idx %= chunk_strides[d];

                let global_coord = offsets[d] as usize + coord_in_chunk;
                if global_coord >= shape[d] as usize {
                    out_of_bounds = true;
                    break;
                }
                ds_flat += global_coord * ds_strides[d];
            }

            if out_of_bounds {
                // Zero-filled (already initialized)
                continue;
            }

            let src_start = ds_flat * element_size;
            let dst_start = flat_idx * element_size;

            if src_start + element_size <= raw_data.len() {
                chunk_bytes[dst_start..dst_start + element_size]
                    .copy_from_slice(&raw_data[src_start..src_start + element_size]);
            }
        }

        result.push((offsets, chunk_bytes));
    }

    result
}

/// Parallel compression threshold: use rayon when chunk count exceeds this.
const PARALLEL_COMPRESS_THRESHOLD: usize = 4;

/// Compress all chunks, using parallel compression when beneficial.
fn compress_all_chunks(
    chunks: &[(Vec<u64>, Vec<u8>)],
    pipeline: &Option<FilterPipeline>,
    element_size: u32,
) -> Result<Vec<Vec<u8>>, FormatError> {
    #[cfg(feature = "parallel")]
    {
        if let Some(ref pl) = pipeline {
            if chunks.len() > PARALLEL_COMPRESS_THRESHOLD {
                use rayon::prelude::*;
                return chunks
                    .par_iter()
                    .map(|(_offsets, chunk_bytes)| {
                        compress_chunk(chunk_bytes, pl, element_size)
                    })
                    .collect();
            }
        }
    }

    // Sequential fallback
    chunks
        .iter()
        .map(|(_offsets, chunk_bytes)| {
            if let Some(ref pl) = pipeline {
                compress_chunk(chunk_bytes, pl, element_size)
            } else {
                Ok(chunk_bytes.clone())
            }
        })
        .collect()
}

/// Build the complete chunked dataset blob (chunk data + index) and return
/// layout/pipeline messages. `base_address` is where the blob will be placed in the file.
/// Serialize a v4 single chunk layout message (public for OH size estimation).
pub fn serialize_v4_single_chunk_pub(
    chunk_dims: &[u32],
    chunk_address: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    serialize_v4_single_chunk(
        chunk_dims, chunk_address, filtered_size, filter_mask, offset_size, element_size,
    )
}

/// Serialize a v4 single chunk layout message.
fn serialize_v4_single_chunk(
    chunk_dims: &[u32],
    chunk_address: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    // flags: bit 0 = unknown meaning in some files, bit 1 = filters for single chunk
    let flags: u8 = if filtered_size.is_some() { 0x02 } else { 0x00 };
    buf.push(flags);

    // dimensionality = rank + 1 (chunk dims + element size dim)
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    // dim_size_encoded_length: how many bytes per dimension
    // We need to figure out the minimum encoding width
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

    // dimension sizes (chunk dims + element size)
    for &d in chunk_dims {
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    // Element size dimension
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 1 (single chunk)
    buf.push(1);

    // Index-specific fields
    if let (Some(fs), Some(fm)) = (filtered_size, filter_mask) {
        // filtered_size (length_size bytes)
        buf.extend_from_slice(&fs.to_le_bytes()); // 8 bytes for length_size=8
        buf.extend_from_slice(&fm.to_le_bytes()); // 4 bytes
    }

    // chunk address
    match offset_size {
        4 => buf.extend_from_slice(&(chunk_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Serialize a v4 Fixed Array layout message.
fn serialize_v4_fixed_array(
    chunk_dims: &[u32],
    fixed_array_address: u64,
    offset_size: u8,
    element_size: u32,
    max_bits: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    let flags: u8 = 0x00;
    buf.push(flags);

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

    // chunk index type = 3 (Fixed Array)
    buf.push(3);

    // max_dblk_page_nelmts_bits — must match FAHD max_nelmts_bits
    buf.push(max_bits);

    // Fixed Array header address
    match offset_size {
        4 => buf.extend_from_slice(&(fixed_array_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&fixed_array_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Build a complete Fixed Array at a known absolute address.
pub fn build_fixed_array_at(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    fa_base_address: u64,
) -> Vec<u8> {
    let os = offset_size as usize;
    let num_elements = chunks.len();

    // For filtered chunks, compute chunk_size encoding width.
    // Must match the HDF5 C library's H5D_FARRAY_FILT_COMPUTE_CHUNK_SIZE_LEN macro:
    //   chunk_size_len = 1 + ((H5VM_log2_gen(chunk.size) + 8) / 8)
    // where chunk.size is the unfiltered chunk size in bytes (product of all chunk dims).
    let chunk_size_bytes: usize = if has_filters {
        let max_raw = chunks.iter().map(|c| c.raw_size).max().unwrap_or(1);
        let log2_val = if max_raw <= 1 { 0 } else { 63 - max_raw.leading_zeros() };
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

    // FAHD total size
    let nelmts_field_size = length_size as usize;
    let fahd_total_size = 4 + 1 + 1 + 1 + 1 + nelmts_field_size + os + 4;
    let fadb_address = fa_base_address + fahd_total_size as u64;

    // Build FAHD
    let mut fahd = Vec::with_capacity(fahd_total_size);
    fahd.extend_from_slice(b"FAHD");
    fahd.push(0); // version
    fahd.push(client_id);
    fahd.push(elem_size as u8);

    // max_nelmts_bits: use 10 as default (page_size = 1024), matching h5py convention
    let max_bits: u8 = 10;
    fahd.push(max_bits);

    match length_size {
        4 => fahd.extend_from_slice(&(num_elements as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
        _ => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
    }

    match offset_size {
        4 => fahd.extend_from_slice(&(fadb_address as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
        _ => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
    }

    // Checksum
    let checksum = jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    assert_eq!(fahd.len(), fahd_total_size);

    // Build FADB
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);

    // header address
    match offset_size {
        4 => fadb.extend_from_slice(&(fa_base_address as u32).to_le_bytes()),
        8 => fadb.extend_from_slice(&fa_base_address.to_le_bytes()),
        _ => fadb.extend_from_slice(&fa_base_address.to_le_bytes()),
    }

    // Element data
    for chunk in chunks {
        match offset_size {
            4 => fadb.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
            8 => fadb.extend_from_slice(&chunk.address.to_le_bytes()),
            _ => fadb.extend_from_slice(&chunk.address.to_le_bytes()),
        }
        if has_filters {
            // Write compressed size using chunk_size_bytes (variable width)
            let cs_bytes = chunk.compressed_size.to_le_bytes();
            fadb.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
            fadb.extend_from_slice(&chunk.filter_mask.to_le_bytes());
        }
    }

    // FADB checksum
    let fadb_checksum = jenkins_lookup3(&fadb);
    fadb.extend_from_slice(&fadb_checksum.to_le_bytes());

    let mut combined = fahd;
    combined.extend_from_slice(&fadb);
    combined
}

/// Build chunked data with absolute addresses.
/// If `maxshape` has unlimited dims, uses Extensible Array index.
pub fn build_chunked_data_at(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
    options: &ChunkOptions,
    base_address: u64,
) -> Result<ChunkedDataResult, FormatError> {
    build_chunked_data_at_ext(raw_data, shape, chunk_dims, element_size, options, base_address, None)
}

/// Build chunked data with absolute addresses and optional maxshape.
pub fn build_chunked_data_at_ext(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
    options: &ChunkOptions,
    base_address: u64,
    maxshape: Option<&[u64]>,
) -> Result<ChunkedDataResult, FormatError> {
    let pipeline = options.build_pipeline(element_size as u32);

    let chunks = split_into_chunks(raw_data, shape, chunk_dims, element_size);
    let num_chunks = chunks.len();
    let has_filters = pipeline.is_some();

    // Compress each chunk (parallel when rayon available and chunk count > 4)
    let compressed_chunks = compress_all_chunks(&chunks, &pipeline, element_size as u32)?;

    let mut data_buf = Vec::new();
    let mut written_chunks = Vec::with_capacity(num_chunks);

    for ((_offsets, chunk_bytes), compressed) in chunks.iter().zip(compressed_chunks) {
        let address = base_address + data_buf.len() as u64;
        let compressed_size = compressed.len() as u64;
        let raw_size = chunk_bytes.len() as u64;

        data_buf.extend_from_slice(&compressed);

        written_chunks.push(WrittenChunk {
            address,
            compressed_size,
            raw_size,
            filter_mask: 0,
        });
    }

    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();
    let offset_size: u8 = 8;
    let length_size: u8 = 8;

    // Determine if we should use Extensible Array (resizable datasets)
    let use_extensible = maxshape.is_some_and(|ms| ms.contains(&u64::MAX));

    let layout_message = if use_extensible {
        let ea_address = base_address + data_buf.len() as u64;

        let ea_bytes = ea_writer::build_extensible_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            ea_address,
        );
        data_buf.extend_from_slice(&ea_bytes);

        ea_writer::serialize_v4_extensible_array(
            &chunk_dims_u32,
            ea_address,
            offset_size,
            element_size as u32,
        )
    } else if num_chunks == 1 {
        let chunk_addr = written_chunks[0].address;
        let filtered_size = if has_filters {
            Some(written_chunks[0].compressed_size)
        } else {
            None
        };
        let filter_mask = if has_filters { Some(0u32) } else { None };
        serialize_v4_single_chunk(
            &chunk_dims_u32,
            chunk_addr,
            filtered_size,
            filter_mask,
            offset_size,
            element_size as u32,
        )
    } else {
        let fa_address = base_address + data_buf.len() as u64;
        let max_bits: u8 = 10;

        let fa_bytes = build_fixed_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            fa_address,
        );
        data_buf.extend_from_slice(&fa_bytes);

        serialize_v4_fixed_array(
            &chunk_dims_u32,
            fa_address,
            offset_size,
            element_size as u32,
            max_bits,
        )
    };

    let pipeline_message = pipeline.as_ref().map(|pl| pl.serialize());

    Ok(ChunkedDataResult {
        data_bytes: data_buf,
        layout_message,
        pipeline_message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_read::read_chunked_data;
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{Datatype, DatatypeByteOrder};

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Helper: build a chunked file blob and read it back using read_chunked_data
    fn roundtrip_chunked(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        options: &ChunkOptions,
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let base_address = 0x1000u64;
        let result =
            build_chunked_data_at(&raw, shape, chunk_dims, 8, options, base_address).unwrap();

        // Build a fake file buffer
        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        // Parse layout
        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        // Parse pipeline if present
        let pipeline = result
            .pipeline_message
            .as_ref()
            .map(|pm| crate::filter_pipeline::FilterPipeline::parse(pm).unwrap());

        let output = read_chunked_data(
            &file_data,
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            8,
            8,
        )
        .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn split_1d_single_chunk() {
        let data = f64_to_bytes(&[1.0, 2.0, 3.0]);
        let result = split_into_chunks(&data, &[3], &[3], 8);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![0]);
        assert_eq!(bytes_to_f64(&result[0].1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_multiple_chunks() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[10], &[4], 8);
        assert_eq!(result.len(), 3); // ceil(10/4) = 3
        assert_eq!(result[0].0, vec![0]);
        assert_eq!(result[1].0, vec![4]);
        assert_eq!(result[2].0, vec![8]);
        assert_eq!(bytes_to_f64(&result[0].1), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(bytes_to_f64(&result[1].1), vec![4.0, 5.0, 6.0, 7.0]);
        // Last chunk: 2 valid + 2 padding zeros
        assert_eq!(bytes_to_f64(&result[2].1), vec![8.0, 9.0, 0.0, 0.0]);
    }

    #[test]
    fn split_2d_chunks() {
        // 4x4 dataset, 2x2 chunks -> 4 chunks
        let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[4, 4], &[2, 2], 8);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, vec![0, 0]);
        assert_eq!(result[1].0, vec![0, 2]);
        assert_eq!(result[2].0, vec![2, 0]);
        assert_eq!(result[3].0, vec![2, 2]);
        // chunk (0,0): elements [0,1,4,5]
        assert_eq!(bytes_to_f64(&result[0].1), vec![0.0, 1.0, 4.0, 5.0]);
        // chunk (0,2): elements [2,3,6,7]
        assert_eq!(bytes_to_f64(&result[1].1), vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn roundtrip_1d_single_chunk_no_compression() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![10]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[10], &[10], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_single_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![100]),
            deflate_level: Some(6),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[100], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_1d_multi_chunk_no_compression() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![8]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[20], &[8], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_multi_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            deflate_level: Some(6),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_shuffle_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![50]),
            deflate_level: Some(6),
            shuffle: true,
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[50], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_2d_chunks() {
        // 6x4 dataset, 3x2 chunks
        let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![3, 2]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[6, 4], &[3, 2], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn chunk_options_auto_dims() {
        let options = ChunkOptions {
            chunk_dims: None,
            deflate_level: Some(6),
            ..Default::default()
        };
        let dims = options.resolve_chunk_dims(&[100, 50]);
        assert_eq!(dims, vec![100, 50]);
    }

    #[test]
    fn chunk_options_pipeline_deflate() {
        let options = ChunkOptions {
            deflate_level: Some(6),
            ..Default::default()
        };
        let pl = options.build_pipeline(8).unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_DEFLATE);
    }

    #[test]
    fn chunk_options_pipeline_lz4() {
        let options = ChunkOptions {
            lz4: true,
            ..Default::default()
        };
        let pl = options.build_pipeline(8).unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_LZ4);
    }

    #[test]
    fn chunk_options_pipeline_zstd() {
        let options = ChunkOptions {
            zstd_level: Some(3),
            ..Default::default()
        };
        let pl = options.build_pipeline(8).unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_ZSTD);
        assert_eq!(pl.filters[0].client_data, vec![3]);
    }

    #[test]
    fn chunk_options_zstd_priority_over_deflate() {
        let options = ChunkOptions {
            deflate_level: Some(6),
            zstd_level: Some(3),
            ..Default::default()
        };
        let pl = options.build_pipeline(8).unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_ZSTD);
    }

    #[test]
    fn chunk_options_pipeline_shuffle_deflate_fletcher32() {
        let options = ChunkOptions {
            deflate_level: Some(6),
            shuffle: true,
            fletcher32: true,
            ..Default::default()
        };
        let pl = options.build_pipeline(8).unwrap();
        assert_eq!(pl.filters.len(), 3);
        assert_eq!(pl.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(pl.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(pl.filters[2].filter_id, FILTER_FLETCHER32);
    }

    #[test]
    fn serialize_v4_single_chunk_no_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[20], 0x1000, None, None, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                btree_address,
                version,
                chunk_index_type,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(1));
                assert_eq!(chunk_dimensions, vec![20, 8]);
                assert_eq!(btree_address, Some(0x1000));
                assert_eq!(single_chunk_filtered_size, None);
                assert_eq!(single_chunk_filter_mask, None);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_single_chunk_with_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[100], 0x2000, Some(500), Some(0), 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                btree_address,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
                ..
            } => {
                assert_eq!(btree_address, Some(0x2000));
                assert_eq!(single_chunk_filtered_size, Some(500));
                assert_eq!(single_chunk_filter_mask, Some(0));
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_fixed_array_roundtrip() {
        let msg = serialize_v4_fixed_array(&[20], 0x3000, 8, 8, 4);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(3));
                assert_eq!(btree_address, Some(0x3000));
                assert_eq!(chunk_dimensions, vec![20, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_fixed_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 160,
                raw_size: 160,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x10A0,
                compressed_size: 160,
                raw_size: 160,
                filter_mask: 0,
            },
        ];
        let fa = build_fixed_array_at(&chunks, 8, 8, false, 0x2000);
        // Should start with FAHD
        assert_eq!(&fa[0..4], b"FAHD");
        // FAHD size = 4+1+1+1+1+8+8+4 = 28
        // FADB starts at offset 28
        assert_eq!(&fa[28..32], b"FADB");
    }

    // ---- Extensible Array tests ----

    #[test]
    fn serialize_v4_extensible_array_roundtrip() {
        let msg = ea_writer::serialize_v4_extensible_array(&[10], 0x4000, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(4));
                assert_eq!(btree_address, Some(0x4000));
                assert_eq!(chunk_dimensions, vec![10, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_extensible_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 80,
                raw_size: 80,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x1050,
                compressed_size: 80,
                raw_size: 80,
                filter_mask: 0,
            },
        ];
        let ea = ea_writer::build_extensible_array_at(&chunks, 8, 8, false, 0x2000);
        assert_eq!(&ea[0..4], b"EAHD");
        // Find EAIB after EAHD: 12 fixed + 6*8 stats + 8 addr + 4 checksum = 72
        let aehd_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
        assert_eq!(&ea[aehd_size..aehd_size + 4], b"EAIB");
    }

    /// Helper: roundtrip with EA (maxshape)
    fn roundtrip_ea(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        maxshape: &[u64],
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let base_address = 0x1000u64;
        let options = ChunkOptions {
            chunk_dims: Some(chunk_dims.to_vec()),
            ..Default::default()
        };
        let result = build_chunked_data_at_ext(
            &raw, shape, chunk_dims, 8, &options, base_address, Some(maxshape),
        )
        .unwrap();

        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        // Verify it uses EA index
        match &layout {
            DataLayout::Chunked { chunk_index_type, .. } => {
                assert_eq!(*chunk_index_type, Some(4), "expected EA index type");
            }
            _ => panic!("expected chunked layout"),
        }

        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: Some(maxshape.to_vec()),
        };
        let datatype = make_f64_type();

        let output = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn ea_roundtrip_1d_inline_only() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[10], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_multi_chunks() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[20], &[5], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_many_chunks() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[100], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    // ---- h5py round-trip tests for chunked writes ----

    #[cfg(feature = "std")]
    fn h5py_run(script: &str) -> String {
        let o = std::process::Command::new("python3").args(["-c", script]).output().expect("python3");
        if !o.status.success() { panic!("h5py: {}", String::from_utf8_lossy(&o.stderr)); }
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_multiple_chunked_datasets() {
        use crate::file_writer::{FileWriter, AttrValue};
        let mut fw = FileWriter::new();
        let data1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data2: Vec<f64> = (0..30).map(|i| (i * 10) as f64).collect();
        fw.create_dataset("a").with_f64_data(&data1).with_shape(&[50]).with_chunks(&[25]);
        fw.create_dataset("b").with_f64_data(&data2).with_shape(&[30]).with_chunks(&[10]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_multi.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); print(json.dumps({{'a':f['a'][:].tolist(),'b':f['b'][:].tolist()}}))",
            path.display()
        );
        let v: serde_json::Value = serde_json::from_str(&h5py_run(&script)).unwrap();
        let va: Vec<f64> = serde_json::from_value(v["a"].clone()).unwrap();
        let vb: Vec<f64> = serde_json::from_value(v["b"].clone()).unwrap();
        assert_eq!(va, data1);
        assert_eq!(vb, data2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_with_attrs() {
        use crate::file_writer::{FileWriter, AttrValue};
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        fw.create_dataset("data").with_f64_data(&data).with_shape(&[50]).with_chunks(&[25])
            .set_attr("units", AttrValue::String("meters".to_string()));
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'units':d.attrs['units'].decode() if isinstance(d.attrs['units'],bytes) else str(d.attrs['units'])}}))",
            path.display()
        );
        let v: serde_json::Value = serde_json::from_str(&h5py_run(&script)).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["units"], serde_json::json!("meters"));
    }

    // --- LZ4 chunked roundtrip tests ---

    #[cfg(feature = "lz4")]
    #[test]
    fn roundtrip_1d_single_chunk_lz4() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![100]),
            lz4: true,
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[100], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn roundtrip_1d_multi_chunk_lz4() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            lz4: true,
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    // --- Zstd chunked roundtrip tests ---

    #[cfg(feature = "zstd")]
    #[test]
    fn roundtrip_1d_single_chunk_zstd() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![100]),
            zstd_level: Some(3),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[100], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn roundtrip_1d_multi_chunk_zstd() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            zstd_level: Some(1),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn roundtrip_1d_shuffle_zstd() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![50]),
            zstd_level: Some(3),
            shuffle: true,
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[50], &options);
        assert_eq!(result, values);
    }
}
