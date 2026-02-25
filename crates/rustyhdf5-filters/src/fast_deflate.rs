//! Fast deflate backends: Apple Compression Framework and zlib-ng.
//!
//! Backend selection priority (decompression & compression):
//! 1. Apple Compression Framework (macOS only, `apple-compression` feature)
//! 2. flate2 with zlib-ng backend (`fast-deflate` feature) or miniz_oxide (default)
//!
//! The Apple Compression Framework uses hardware-accelerated zlib on Apple Silicon
//! and is typically the fastest option on macOS. zlib-ng is the fastest portable
//! option and what C HDF5 uses internally.

// ---------------------------------------------------------------------------
// Apple Compression Framework FFI (macOS only)
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "apple-compression"))]
mod apple {
    use std::os::raw::c_int;

    // compression.h constants
    const COMPRESSION_ZLIB: c_int = 0x205;

    // compression.h operations
    const COMPRESSION_STREAM_ENCODE: c_int = 0;
    const COMPRESSION_STREAM_DECODE: c_int = 1;

    // Return codes
    const COMPRESSION_STATUS_OK: c_int = 0;
    const COMPRESSION_STATUS_END: c_int = 1;
    const COMPRESSION_STATUS_ERROR: c_int = -1;

    // Flags
    const COMPRESSION_STREAM_FINALIZE: c_int = 0x0001;

    #[repr(C)]
    struct CompressionStream {
        dst_ptr: *mut u8,
        dst_size: usize,
        src_ptr: *const u8,
        src_size: usize,
        state: *mut std::ffi::c_void,
    }

    #[link(name = "compression")]
    unsafe extern "C" {
        fn compression_stream_init(
            stream: *mut CompressionStream,
            operation: c_int,
            algorithm: c_int,
        ) -> c_int;

        fn compression_stream_process(
            stream: *mut CompressionStream,
            flags: c_int,
        ) -> c_int;

        fn compression_stream_destroy(stream: *mut CompressionStream) -> c_int;
    }

    /// Decompress zlib data using Apple's Compression framework.
    ///
    /// Apple Compression expects raw deflate data, but HDF5/zlib uses the zlib
    /// wrapper format (2-byte header + data + 4-byte checksum). We strip the
    /// zlib wrapper before passing to Apple Compression.
    pub(crate) fn decompress(data: &[u8], output_hint: usize) -> Result<Vec<u8>, String> {
        // Zlib format: [CMF][FLG] [DICTID?] [compressed data] [ADLER32]
        // Strip the 2-byte zlib header and 4-byte adler32 trailer.
        if data.len() < 6 {
            return Err("apple compression: zlib data too short".into());
        }

        let cmf = data[0];
        if cmf & 0x0F != 8 {
            return Err("apple compression: not zlib/deflate data".into());
        }

        // Check for FDICT flag
        let flg = data[1];
        let header_size = if flg & 0x20 != 0 { 6 } else { 2 };
        if data.len() < header_size + 4 {
            return Err("apple compression: zlib data too short for header + trailer".into());
        }

        let raw_deflate = &data[header_size..data.len() - 4];

        let capacity = if output_hint > 0 { output_hint } else { data.len() * 4 };
        let mut output = vec![0u8; capacity];
        let mut total_written = 0usize;

        unsafe {
            let mut stream = std::mem::zeroed::<CompressionStream>();
            let status = compression_stream_init(
                &mut stream,
                COMPRESSION_STREAM_DECODE,
                COMPRESSION_ZLIB,
            );
            if status != COMPRESSION_STATUS_OK {
                return Err("apple compression: failed to init decode stream".into());
            }

            stream.src_ptr = raw_deflate.as_ptr();
            stream.src_size = raw_deflate.len();

            loop {
                stream.dst_ptr = output.as_mut_ptr().add(total_written);
                stream.dst_size = output.len() - total_written;

                let result = compression_stream_process(&mut stream, COMPRESSION_STREAM_FINALIZE);
                total_written = output.len() - stream.dst_size;

                match result {
                    COMPRESSION_STATUS_END => {
                        compression_stream_destroy(&mut stream);
                        output.truncate(total_written);
                        return Ok(output);
                    }
                    COMPRESSION_STATUS_OK => {
                        // Need more output space
                        if stream.dst_size == 0 {
                            output.resize(output.len() * 2, 0);
                        } else {
                            // OK with remaining src=0 means done
                            if stream.src_size == 0 {
                                compression_stream_destroy(&mut stream);
                                output.truncate(total_written);
                                return Ok(output);
                            }
                        }
                    }
                    COMPRESSION_STATUS_ERROR => {
                        compression_stream_destroy(&mut stream);
                        return Err("apple compression: decompression error".into());
                    }
                    other => {
                        compression_stream_destroy(&mut stream);
                        return Err(format!("apple compression: unexpected status {other}"));
                    }
                }
            }
        }
    }

    /// Compress data using Apple's Compression framework with zlib wrapper.
    pub(crate) fn compress(data: &[u8], _level: u32) -> Result<Vec<u8>, String> {
        if data.is_empty() {
            // Return a valid empty zlib stream
            return Ok(vec![0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        }

        // Compress raw deflate data, then wrap in zlib format.
        let max_size = data.len() + data.len() / 10 + 64;
        let mut raw_output = vec![0u8; max_size];
        let mut total_written = 0usize;

        unsafe {
            let mut stream = std::mem::zeroed::<CompressionStream>();
            let status = compression_stream_init(
                &mut stream,
                COMPRESSION_STREAM_ENCODE,
                COMPRESSION_ZLIB,
            );
            if status != COMPRESSION_STATUS_OK {
                return Err("apple compression: failed to init encode stream".into());
            }

            stream.src_ptr = data.as_ptr();
            stream.src_size = data.len();

            loop {
                stream.dst_ptr = raw_output.as_mut_ptr().add(total_written);
                stream.dst_size = raw_output.len() - total_written;

                let result = compression_stream_process(&mut stream, COMPRESSION_STREAM_FINALIZE);
                total_written = raw_output.len() - stream.dst_size;

                match result {
                    COMPRESSION_STATUS_END => break,
                    COMPRESSION_STATUS_OK => {
                        if stream.dst_size == 0 {
                            raw_output.resize(raw_output.len() * 2, 0);
                        } else if stream.src_size == 0 {
                            break;
                        }
                    }
                    _ => {
                        compression_stream_destroy(&mut stream);
                        return Err("apple compression: compression error".into());
                    }
                }
            }

            compression_stream_destroy(&mut stream);
            raw_output.truncate(total_written);
        }

        // Build zlib-wrapped output: header + raw deflate + adler32
        let mut output = Vec::with_capacity(2 + raw_output.len() + 4);
        // Zlib header: CM=8 (deflate), CINFO=7 (32K window), FCHECK to make header % 31 == 0
        let cmf: u8 = 0x78;
        let flg: u8 = 0x9C; // level=default, no dict, FCHECK=28
        output.push(cmf);
        output.push(flg);
        output.extend_from_slice(&raw_output);
        // Adler32 checksum of uncompressed data
        let adler = adler32(data);
        output.extend_from_slice(&adler.to_be_bytes());

        Ok(output)
    }

    /// Compute Adler-32 checksum.
    fn adler32(data: &[u8]) -> u32 {
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        // Process in blocks of 5552 to avoid overflow with u32 accumulators.
        const BLOCK: usize = 5552;
        for chunk in data.chunks(BLOCK) {
            for &byte in chunk {
                a += byte as u32;
                b += a;
            }
            a %= 65521;
            b %= 65521;
        }
        (b << 16) | a
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn apple_adler32_empty() {
            assert_eq!(adler32(&[]), 1);
        }

        #[test]
        fn apple_adler32_known() {
            // "Wikipedia" -> 0x11E60398
            assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming decompression via flate2 (uses zlib-ng when fast-deflate enabled)
// ---------------------------------------------------------------------------

/// Streaming decompress with pre-allocated output buffer.
///
/// When the output size is known (typical for HDF5 chunks), this avoids
/// dynamic reallocation by writing directly into a pre-sized buffer.
pub(crate) fn flate2_decompress_preallocated(
    data: &[u8],
    output_size: usize,
) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut output = vec![0u8; output_size];
    let mut total_read = 0;

    loop {
        match decoder.read(&mut output[total_read..]) {
            Ok(0) => break,
            Ok(n) => total_read += n,
            Err(e) => return Err(e.to_string()),
        }
    }
    output.truncate(total_read);
    Ok(output)
}

/// Streaming decompress with dynamic sizing (when output size is unknown).
pub(crate) fn flate2_decompress_streaming(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    decoder.read_to_end(&mut result).map_err(|e| e.to_string())?;
    Ok(result)
}

/// Compress data using flate2 (zlib-ng when fast-deflate enabled, else miniz_oxide).
pub(crate) fn flate2_compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder.write_all(data).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Public API: backend selection
// ---------------------------------------------------------------------------

/// Decompress zlib data using the fastest available backend.
///
/// Selection order:
/// 1. Apple Compression Framework (macOS + `apple-compression` feature)
/// 2. flate2 (zlib-ng with `fast-deflate`, otherwise miniz_oxide)
///
/// When `output_hint` > 0, pre-allocates the output buffer for zero-copy
/// decompression (avoids reallocation).
pub fn decompress(data: &[u8], output_hint: usize) -> Result<Vec<u8>, String> {
    #[cfg(all(target_os = "macos", feature = "apple-compression"))]
    {
        match apple::decompress(data, output_hint) {
            Ok(result) => return Ok(result),
            Err(_) => {} // Fall through to flate2
        }
    }

    if output_hint > 0 {
        flate2_decompress_preallocated(data, output_hint)
    } else {
        flate2_decompress_streaming(data)
    }
}

/// Compress data using the fastest available backend.
///
/// Selection order:
/// 1. Apple Compression Framework (macOS + `apple-compression` feature)
/// 2. flate2 (zlib-ng with `fast-deflate`, otherwise miniz_oxide)
pub fn compress(data: &[u8], level: u32) -> Result<Vec<u8>, String> {
    #[cfg(all(target_os = "macos", feature = "apple-compression"))]
    {
        match apple::compress(data, level) {
            Ok(result) => return Ok(result),
            Err(_) => {} // Fall through to flate2
        }
    }

    flate2_compress(data, level)
}

/// Returns a human-readable name of the active decompression backend.
pub fn active_backend() -> &'static str {
    #[cfg(all(target_os = "macos", feature = "apple-compression"))]
    {
        return "apple-compression";
    }
    #[cfg(all(not(all(target_os = "macos", feature = "apple-compression")), feature = "fast-deflate"))]
    {
        return "zlib-ng";
    }
    #[cfg(not(any(
        all(target_os = "macos", feature = "apple-compression"),
        feature = "fast-deflate"
    )))]
    {
        "miniz_oxide"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_decompress_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let compressed = compress(&data, 6).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn fast_decompress_no_hint() {
        let data: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let compressed = compress(&data, 6).unwrap();
        let decompressed = decompress(&compressed, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn fast_decompress_empty() {
        let compressed = compress(&[], 6).unwrap();
        let decompressed = decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn fast_decompress_large() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let compressed = compress(&data, 6).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn preallocated_vs_streaming_match() {
        let data: Vec<u8> = (0..2000).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = flate2_compress(&data, 6).unwrap();
        let prealloc = flate2_decompress_preallocated(&compressed, data.len()).unwrap();
        let streaming = flate2_decompress_streaming(&compressed).unwrap();
        assert_eq!(prealloc, streaming);
        assert_eq!(prealloc, data);
    }

    #[test]
    fn backend_name_is_set() {
        let name = active_backend();
        assert!(
            ["miniz_oxide", "zlib-ng", "apple-compression"].contains(&name),
            "unexpected backend: {name}"
        );
    }

    #[test]
    fn cross_backend_with_python_zlib() {
        // python3 -c "import zlib; print(list(zlib.compress(bytes(range(10)), 6)))"
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed = decompress(&compressed, 10).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
