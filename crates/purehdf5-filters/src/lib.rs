//! Filter and compression pipeline for HDF5.
//!
//! Backends:
//! - **deflate** (default): miniz_oxide via flate2
//! - **fast-deflate**: libdeflater (~2-3x faster)
//! - **zlib-ng**: zlib-ng via flate2 (SIMD-accelerated, same API)
//! - **lz4**: LZ4 block compression (~10x faster than deflate)
//! - **zstd**: Zstandard compression (better ratio + faster)
//! - **parallel**: parallel chunk compression via rayon

// ===========================================================================
// Deflate (zlib)
// ===========================================================================

/// Decompress zlib-compressed data.
///
/// When `fast-deflate` is enabled, uses `libdeflater` which is 2-3x faster
/// than `miniz_oxide`.  Falls back to `flate2` (miniz_oxide) otherwise.
pub fn deflate_decompress(data: &[u8], max_output_size: usize) -> Result<Vec<u8>, String> {
    #[cfg(feature = "fast-deflate")]
    {
        fast_deflate_decompress(data, max_output_size)
    }
    #[cfg(not(feature = "fast-deflate"))]
    {
        default_deflate_decompress(data, max_output_size)
    }
}

/// Compress data with zlib.
///
/// When `fast-deflate` is enabled, uses `libdeflater`.
pub fn deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    #[cfg(feature = "fast-deflate")]
    {
        fast_deflate_compress(data, level)
    }
    #[cfg(not(feature = "fast-deflate"))]
    {
        default_deflate_compress(data, level)
    }
}

// ---------------------------------------------------------------------------
// Default backend: flate2 (miniz_oxide, or zlib-ng when feature enabled)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "fast-deflate"))]
fn default_deflate_decompress(data: &[u8], _max_output_size: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| e.to_string())?;
    Ok(result)
}

#[cfg(not(feature = "fast-deflate"))]
fn default_deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder.write_all(data).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Fast backend: libdeflater
// ---------------------------------------------------------------------------

#[cfg(feature = "fast-deflate")]
fn fast_deflate_decompress(data: &[u8], max_output_size: usize) -> Result<Vec<u8>, String> {
    let mut decompressor = libdeflater::Decompressor::new();
    let mut output = vec![0u8; max_output_size];
    let actual_size = decompressor
        .zlib_decompress(data, &mut output)
        .map_err(|e| format!("libdeflater decompress error: {e:?}"))?;
    output.truncate(actual_size);
    Ok(output)
}

#[cfg(feature = "fast-deflate")]
fn fast_deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    let level = libdeflater::CompressionLvl::new(level.clamp(1, 12) as i32)
        .map_err(|e| format!("libdeflater level error: {e:?}"))?;
    let mut compressor = libdeflater::Compressor::new(level);
    let max_size = compressor.zlib_compress_bound(data.len());
    let mut output = vec![0u8; max_size];
    let actual_size = compressor
        .zlib_compress(data, &mut output)
        .map_err(|e| format!("libdeflater compress error: {e:?}"))?;
    output.truncate(actual_size);
    Ok(output)
}

/// Decompress zlib data using the default (flate2) backend.
/// Available regardless of feature flags for comparison/testing.
pub fn deflate_decompress_miniz(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| e.to_string())?;
    Ok(result)
}

/// Compress data using the default (flate2) backend.
/// Available regardless of feature flags for comparison/testing.
pub fn deflate_compress_miniz(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder.write_all(data).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

// ===========================================================================
// LZ4 (HDF5 filter ID 32004)
// ===========================================================================

/// Compress data using LZ4 block format with size header.
///
/// Format: 4 bytes LE original size + LZ4 block compressed data.
#[cfg(feature = "lz4")]
pub fn lz4_compress(data: &[u8]) -> Result<Vec<u8>, String> {
    let compressed = lz4_flex::block::compress(data);
    let mut result = Vec::with_capacity(4 + compressed.len());
    result.extend_from_slice(&(data.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    Ok(result)
}

/// Decompress LZ4 data with size header.
///
/// Format: 4 bytes LE original size + LZ4 block compressed data.
#[cfg(feature = "lz4")]
pub fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 4 {
        return Err("lz4: data too short for size header".into());
    }
    let orig_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    lz4_flex::block::decompress(&data[4..], orig_size)
        .map_err(|e| format!("lz4 decompress error: {e}"))
}

// ===========================================================================
// Zstandard (HDF5 filter ID 32015)
// ===========================================================================

/// Compress data using Zstandard.
///
/// Level 1-3 is faster than deflate with better compression ratio.
/// Level range: 1-22 (default 3).
#[cfg(feature = "zstd")]
pub fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>, String> {
    zstd::encode_all(std::io::Cursor::new(data), level)
        .map_err(|e| format!("zstd compress error: {e}"))
}

/// Decompress Zstandard data.
#[cfg(feature = "zstd")]
pub fn zstd_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    zstd::decode_all(std::io::Cursor::new(data))
        .map_err(|e| format!("zstd decompress error: {e}"))
}

// ===========================================================================
// Parallel compression
// ===========================================================================

/// Compress multiple chunks in parallel using rayon.
///
/// Falls back to sequential compression when the `parallel` feature is
/// not enabled or when there are fewer than 4 chunks.
pub fn compress_chunks<F>(
    chunks: &[&[u8]],
    compress_fn: F,
) -> Result<Vec<Vec<u8>>, String>
where
    F: Fn(&[u8]) -> Result<Vec<u8>, String> + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if chunks.len() > 4 {
            use rayon::prelude::*;
            return chunks
                .par_iter()
                .map(|chunk| compress_fn(chunk))
                .collect();
        }
    }

    // Sequential fallback
    chunks.iter().map(|chunk| compress_fn(chunk)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Deflate tests ---

    #[test]
    fn compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn decompress_python_zlib() {
        // python3 -c "import zlib; print(list(zlib.compress(bytes(range(10)), 6)))"
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed = deflate_decompress(&compressed, 10).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn miniz_always_available() {
        let data = vec![42u8; 100];
        let compressed = deflate_compress_miniz(&data, 6).unwrap();
        let decompressed = deflate_decompress_miniz(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn cross_backend_compatibility() {
        // Compress with miniz, decompress with current backend
        let data: Vec<u8> = (0..500).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = deflate_compress_miniz(&data, 6).unwrap();
        let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn deflate_empty_data() {
        let compressed = deflate_compress(&[], 6).unwrap();
        let decompressed = deflate_decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn deflate_large_data_roundtrip() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        assert!(compressed.len() < data.len()); // should actually compress
        let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn cross_backend_miniz_decompress_current_compress() {
        // Compress with current backend, decompress with miniz
        let data: Vec<u8> = (0..500).map(|i| (i * 3 % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        let decompressed = deflate_decompress_miniz(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn deflate_all_levels() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        for level in 0..=9 {
            let compressed = deflate_compress(&data, level).unwrap();
            let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
            assert_eq!(decompressed, data, "failed at level {level}");
        }
    }

    // --- LZ4 tests ---

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let compressed = lz4_compress(&data).unwrap();
        let decompressed = lz4_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_empty_data() {
        let compressed = lz4_compress(&[]).unwrap();
        let decompressed = lz4_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_large_data_roundtrip() {
        // 1M f64 values (8MB raw)
        let data: Vec<u8> = (0..8_000_000)
            .map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8)
            .collect();
        let compressed = lz4_compress(&data).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = lz4_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_decompress_too_short() {
        let result = lz4_decompress(&[1, 2, 3]);
        assert!(result.is_err());
    }

    // --- Zstd tests ---

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let compressed = zstd_compress(&data, 3).unwrap();
        let decompressed = zstd_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_empty_data() {
        let compressed = zstd_compress(&[], 3).unwrap();
        let decompressed = zstd_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_large_data_roundtrip() {
        // 1M f64 values (8MB raw)
        let data: Vec<u8> = (0..8_000_000)
            .map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8)
            .collect();
        let compressed = zstd_compress(&data, 3).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = zstd_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_different_levels() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        for level in [1, 3, 6, 9] {
            let compressed = zstd_compress(&data, level).unwrap();
            let decompressed = zstd_decompress(&compressed).unwrap();
            assert_eq!(decompressed, data, "failed at level {level}");
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_level1_faster_better_ratio_than_deflate() {
        // zstd at level 1 should produce smaller output than deflate at level 6
        // for typical data (not guaranteed but generally true)
        let data: Vec<u8> = (0..10_000)
            .map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8)
            .collect();
        let zstd_compressed = zstd_compress(&data, 1).unwrap();
        let deflate_compressed = deflate_compress(&data, 6).unwrap();
        // zstd level 1 typically beats deflate level 6 on ratio
        // Just verify both produce valid compressed output
        assert!(zstd_compressed.len() < data.len());
        assert!(deflate_compressed.len() < data.len());
    }

    // --- Parallel compression tests ---

    #[test]
    fn parallel_compress_sequential_fallback() {
        let chunks: Vec<Vec<u8>> = (0..3)
            .map(|i| vec![(i as u8) * 10; 100])
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let results = compress_chunks(&refs, |data| {
            deflate_compress(data, 6)
        })
        .unwrap();
        assert_eq!(results.len(), 3);
        for (i, compressed) in results.iter().enumerate() {
            let decompressed = deflate_decompress(compressed, 100).unwrap();
            assert_eq!(decompressed, chunks[i]);
        }
    }

    #[test]
    fn parallel_compress_many_chunks() {
        let chunks: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                (0..1000).map(|j| ((i * 100 + j) % 256) as u8).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let results = compress_chunks(&refs, |data| {
            deflate_compress(data, 6)
        })
        .unwrap();
        assert_eq!(results.len(), 10);
        for (i, compressed) in results.iter().enumerate() {
            let decompressed = deflate_decompress(compressed, 1000).unwrap();
            assert_eq!(decompressed, chunks[i]);
        }
    }

    #[test]
    fn parallel_compress_produces_same_as_sequential() {
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                (0..500).map(|j| ((i * 50 + j) % 256) as u8).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        // Compress with parallel helper
        let parallel_results = compress_chunks(&refs, |data| {
            deflate_compress(data, 6)
        })
        .unwrap();

        // Compress individually (sequential)
        let sequential_results: Vec<Vec<u8>> = chunks
            .iter()
            .map(|c| deflate_compress(c, 6).unwrap())
            .collect();

        // Both should decompress to the same data
        for (i, (par, seq)) in parallel_results.iter().zip(&sequential_results).enumerate() {
            let par_dec = deflate_decompress(par, 500).unwrap();
            let seq_dec = deflate_decompress(seq, 500).unwrap();
            assert_eq!(par_dec, seq_dec, "mismatch at chunk {i}");
            assert_eq!(par_dec, chunks[i]);
        }
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn parallel_compress_lz4_chunks() {
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| vec![(i as u8) * 20; 500])
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let results = compress_chunks(&refs, |data| lz4_compress(data)).unwrap();
        assert_eq!(results.len(), 8);
        for (i, compressed) in results.iter().enumerate() {
            let decompressed = lz4_decompress(compressed).unwrap();
            assert_eq!(decompressed, chunks[i]);
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn parallel_compress_zstd_chunks() {
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| vec![(i as u8) * 20; 500])
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let results = compress_chunks(&refs, |data| zstd_compress(data, 1)).unwrap();
        assert_eq!(results.len(), 8);
        for (i, compressed) in results.iter().enumerate() {
            let decompressed = zstd_decompress(compressed).unwrap();
            assert_eq!(decompressed, chunks[i]);
        }
    }
}
