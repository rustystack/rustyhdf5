//! Filter and compression pipeline for HDF5.
//!
//! Provides deflate (zlib) decompression/compression with an optional fast
//! backend via the `fast-deflate` feature (uses `libdeflater` instead of
//! `miniz_oxide`).

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
// Default backend: flate2 (miniz_oxide)
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

/// Decompress zlib data using the default (miniz_oxide) backend.
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

/// Compress data using the default (miniz_oxide) backend.
/// Available regardless of feature flags for comparison/testing.
pub fn deflate_compress_miniz(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder.write_all(data).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn empty_data() {
        let compressed = deflate_compress(&[], 6).unwrap();
        let decompressed = deflate_decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn large_data_roundtrip() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        assert!(compressed.len() < data.len()); // should actually compress
        let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
