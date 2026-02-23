//! Filter and compression pipeline for HDF5.
//!
//! Provides deflate (zlib) decompression/compression with multiple backend options:
//!
//! - **Default**: `miniz_oxide` (pure Rust, no C dependencies)
//! - **`fast-deflate` feature**: `zlib-ng` via flate2 (~2-3x faster, matches C HDF5)
//! - **`apple-compression` feature**: Apple Compression Framework on macOS
//!   (hardware-accelerated on Apple Silicon)
//!
//! Backend priority: apple-compression > zlib-ng > miniz_oxide.

pub mod fast_deflate;

/// Decompress zlib-compressed data.
///
/// Uses the fastest available backend. When `max_output_size` > 0,
/// pre-allocates the output buffer for streaming decompression.
pub fn deflate_decompress(data: &[u8], max_output_size: usize) -> Result<Vec<u8>, String> {
    fast_deflate::decompress(data, max_output_size)
}

/// Compress data with zlib.
///
/// Uses the fastest available backend.
pub fn deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    fast_deflate::compress(data, level)
}

/// Decompress zlib data using the pure-Rust miniz_oxide backend.
/// Always available regardless of feature flags, for comparison/testing.
pub fn deflate_decompress_miniz(data: &[u8]) -> Result<Vec<u8>, String> {
    let result = miniz_oxide::inflate::decompress_to_vec_zlib(data)
        .map_err(|e| format!("miniz_oxide decompress error: {e:?}"))?;
    Ok(result)
}

/// Compress data using the pure-Rust miniz_oxide backend.
/// Always available regardless of feature flags, for comparison/testing.
pub fn deflate_compress_miniz(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    let level = level.min(10) as u8;
    let result = miniz_oxide::deflate::compress_to_vec_zlib(data, level);
    Ok(result)
}

/// Returns the name of the currently active deflate backend.
pub fn deflate_backend() -> &'static str {
    fast_deflate::active_backend()
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
        // Compress with miniz, decompress with current (possibly zlib-ng) backend
        let data: Vec<u8> = (0..500).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = deflate_compress_miniz(&data, 6).unwrap();
        let decompressed = deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn cross_backend_reverse() {
        // Compress with current backend, decompress with miniz
        let data: Vec<u8> = (0..500).map(|i| (i * 13 % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        let decompressed = deflate_decompress_miniz(&compressed).unwrap();
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

    #[test]
    fn backend_reports_name() {
        let name = deflate_backend();
        assert!(
            ["miniz_oxide", "zlib-ng", "apple-compression"].contains(&name),
            "unexpected backend: {name}"
        );
    }

    #[test]
    fn all_backends_produce_identical_output() {
        let data: Vec<u8> = (0..10_000).map(|i| (i * 31 % 256) as u8).collect();

        // Compress with current backend
        let compressed_current = deflate_compress(&data, 6).unwrap();
        // Compress with miniz
        let compressed_miniz = deflate_compress_miniz(&data, 6).unwrap();

        // Both should decompress to the same data (even if compressed bytes differ)
        let dec_current = deflate_decompress(&compressed_current, data.len()).unwrap();
        let dec_miniz = deflate_decompress_miniz(&compressed_miniz).unwrap();
        let dec_cross = deflate_decompress_miniz(&compressed_current).unwrap();
        let dec_cross2 = deflate_decompress(&compressed_miniz, data.len()).unwrap();

        assert_eq!(dec_current, data);
        assert_eq!(dec_miniz, data);
        assert_eq!(dec_cross, data);
        assert_eq!(dec_cross2, data);
    }
}
