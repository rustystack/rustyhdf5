//! Tests for zero-copy typed reads of contiguous native-endian datasets.

use rustyhdf5::{AttrValue, Error, File, FileBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_to_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn make_f64_file(values: &[f64]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(values)
        .with_shape(&[values.len() as u64]);
    b.finish().unwrap()
}

fn make_f32_file(values: &[f32]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f32_data(values)
        .with_shape(&[values.len() as u64]);
    b.finish().unwrap()
}

fn make_i32_file(values: &[i32]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_i32_data(values)
        .with_shape(&[values.len() as u64]);
    b.finish().unwrap()
}

fn make_i64_file(values: &[i64]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_i64_data(values)
        .with_shape(&[values.len() as u64]);
    b.finish().unwrap()
}

fn make_u8_file(values: &[u8]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_u8_data(values)
        .with_shape(&[values.len() as u64]);
    b.finish().unwrap()
}

// ---------------------------------------------------------------------------
// Test 1: Zero-copy f64 read matches regular read
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_f64_matches_regular() {
    let values = vec![1.1, 2.2, 3.3, 4.4, 5.5];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_f64_match.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let regular = ds.read_f64().unwrap();
    let zerocopy = ds.read_f64_zerocopy().unwrap();
    assert_eq!(zerocopy, regular.as_slice());
    assert_eq!(zerocopy, &values[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 2: Zero-copy f32 read matches regular read
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_f32_matches_regular() {
    let values = vec![1.5f32, 2.5, 3.5, 4.5];
    let bytes = make_f32_file(&values);
    let path = write_to_temp(&bytes, "zc_test_f32_match.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let regular = ds.read_f32().unwrap();
    let zerocopy = ds.read_f32_zerocopy().unwrap();
    assert_eq!(zerocopy, regular.as_slice());
    assert_eq!(zerocopy, &values[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 3: Zero-copy i32 read matches regular read
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_i32_matches_regular() {
    let values = vec![-10, 0, 42, 100, -999];
    let bytes = make_i32_file(&values);
    let path = write_to_temp(&bytes, "zc_test_i32_match.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let regular = ds.read_i32().unwrap();
    let zerocopy = ds.read_i32_zerocopy().unwrap();
    assert_eq!(zerocopy, regular.as_slice());
    assert_eq!(zerocopy, &values[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 4: Zero-copy i64 read matches regular read
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_i64_matches_regular() {
    let values = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let bytes = make_i64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_i64_match.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let regular = ds.read_i64().unwrap();
    let zerocopy = ds.read_i64_zerocopy().unwrap();
    assert_eq!(zerocopy, regular.as_slice());
    assert_eq!(zerocopy, &values[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 5: Zero-copy u8 read works
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_u8_works() {
    let values: Vec<u8> = vec![0, 1, 127, 255, 42];
    let bytes = make_u8_file(&values);
    let path = write_to_temp(&bytes, "zc_test_u8.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let zerocopy = ds.read_u8_zerocopy().unwrap();
    assert_eq!(zerocopy, &values[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 6: Zero-copy raw bytes read works
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_raw_bytes_works() {
    let values = vec![1.0f64, 2.0, 3.0];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_raw.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let raw = ds.read_raw_zerocopy().unwrap();
    assert_eq!(raw.len(), 3 * 8);
    // Verify first f64 bytes
    assert_eq!(&raw[..8], &1.0f64.to_le_bytes());

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 7: Error on chunked dataset
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_on_chunked() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[1000])
        .with_chunks(&[100]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "zc_test_chunked.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_f64_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyNotContiguous));

    // raw zerocopy also errors
    let err = ds.read_raw_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyNotContiguous));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 8: Error on type mismatch (request f64 on i32 dataset)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_type_mismatch_f64_on_i32() {
    let values = vec![10i32, 20, 30];
    let bytes = make_i32_file(&values);
    let path = write_to_temp(&bytes, "zc_test_type_f64_i32.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_f64_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyTypeMismatch { .. }));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 9: Error on type mismatch (request i32 on f64 dataset)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_type_mismatch_i32_on_f64() {
    let values = vec![1.0f64, 2.0];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_type_i32_f64.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_i32_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyTypeMismatch { .. }));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 10: Error on type mismatch (request u8 on f64 dataset)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_type_mismatch_u8_on_f64() {
    let values = vec![1.0f64, 2.0];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_type_u8_f64.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_u8_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyTypeMismatch { .. }));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 11: Large dataset (1M f64) zero-copy
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_large_1m_f64() {
    let n = 1_000_000;
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&values)
        .with_shape(&[n as u64]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "zc_test_1m.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let zc = ds.read_f64_zerocopy().unwrap();
    assert_eq!(zc.len(), n);
    assert_eq!(zc[0], 0.0);
    assert_eq!(zc[n / 2], (n / 2) as f64);
    assert_eq!(zc[n - 1], (n - 1) as f64);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 12: Verify returned slice points into mmap (not a copy)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_slice_points_into_file_bytes() {
    let values = vec![1.0f64, 2.0, 3.0];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_ptr.h5");

    let file = File::open(&path).unwrap();
    let file_bytes = file.as_bytes();
    let ds = file.dataset("data").unwrap();
    let zc = ds.read_f64_zerocopy().unwrap();

    // The zero-copy slice's memory must lie within the file bytes range.
    let file_start = file_bytes.as_ptr() as usize;
    let file_end = file_start + file_bytes.len();
    let zc_start = zc.as_ptr() as usize;
    let zc_end = zc_start + zc.len() * 8;
    assert!(
        zc_start >= file_start && zc_end <= file_end,
        "zero-copy slice should point into file bytes"
    );

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 13: Round-trip: write f64 dataset, reopen, zero-copy read, verify
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_roundtrip_write_read() {
    let dir = std::env::temp_dir();
    let path = dir.join("zc_roundtrip.h5");

    let original = vec![3.14, 2.718, 1.414, 1.732, 0.577];
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&original)
        .set_attr("unit", AttrValue::String("dimensionless".into()));
    b.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    assert!(file.is_mmap());
    let ds = file.dataset("data").unwrap();
    let zc = ds.read_f64_zerocopy().unwrap();
    assert_eq!(zc, &original[..]);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 14: Zero-copy f32 type mismatch on f64 dataset
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_f32_on_f64() {
    let values = vec![1.0f64, 2.0];
    let bytes = make_f64_file(&values);
    let path = write_to_temp(&bytes, "zc_test_f32_on_f64.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_f32_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyTypeMismatch { .. }));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 15: Zero-copy i64 type mismatch on i32 dataset
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_i64_on_i32() {
    let values = vec![10i32, 20, 30];
    let bytes = make_i32_file(&values);
    let path = write_to_temp(&bytes, "zc_test_i64_on_i32.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_i64_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyTypeMismatch { .. }));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 16: Zero-copy works with from_bytes (owned, not mmap)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_from_bytes_owned() {
    let values = vec![10.0f64, 20.0, 30.0];
    let bytes = make_f64_file(&values);

    let file = File::from_bytes(bytes).unwrap();
    assert!(!file.is_mmap());
    let ds = file.dataset("data").unwrap();
    let zc = ds.read_f64_zerocopy().unwrap();
    assert_eq!(zc, &values[..]);
}

// ---------------------------------------------------------------------------
// Test 17: Zero-copy raw on chunked returns error (not None)
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_raw_chunked_error() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[100])
        .with_chunks(&[50]);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let err = ds.read_raw_zerocopy().unwrap_err();
    assert!(matches!(err, Error::ZeroCopyNotContiguous));
}

// ---------------------------------------------------------------------------
// Test 18: Error display messages
// ---------------------------------------------------------------------------

#[test]
fn zerocopy_error_display() {
    let err = Error::ZeroCopyNotContiguous;
    assert!(err.to_string().contains("contiguous"));

    let err = Error::ZeroCopyNonNativeEndian;
    assert!(err.to_string().contains("native-endian"));

    let err = Error::ZeroCopyTypeMismatch {
        expected: "f64",
        actual: "i32".into(),
    };
    assert!(err.to_string().contains("f64"));
    assert!(err.to_string().contains("i32"));

    let err = Error::ZeroCopyUnaligned {
        required: 8,
        actual: 3,
    };
    assert!(err.to_string().contains("alignment"));
}
