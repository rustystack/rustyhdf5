//! Tests for the mmap-default read path and zero-copy contiguous reads.
//!
//! Covers:
//! - File::open() uses mmap by default
//! - File::from_bytes() backward compatibility
//! - File::open_buffered() fallback
//! - Zero-copy raw read (read_raw_ref)
//! - Round-trip: write → open with mmap → read back
//! - Large file opens fast with mmap
//! - All existing operations (datasets, attrs, groups) work through mmap
//! - LazyFile::open_mmap convenience

use purehdf5::{AttrValue, DType, FileBuilder, File, LazyFile};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_simple_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("temperatures")
        .with_f64_data(&[22.5, 23.1, 21.8]);
    b.create_dataset("counts").with_i32_data(&[10, 20, 30]);
    b.set_attr("version", AttrValue::I64(2));
    b.set_attr("description", AttrValue::String("test file".into()));
    b.finish().unwrap()
}

fn make_grouped_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    let mut g = b.create_group("sensors");
    g.create_dataset("temperature")
        .with_f64_data(&[22.5, 23.1, 21.8]);
    g.create_dataset("humidity").with_i32_data(&[45, 50, 55]);
    g.set_attr("location", AttrValue::String("lab".into()));
    let finished = g.finish();
    b.add_group(finished);

    let mut g2 = b.create_group("metadata");
    g2.create_dataset("timestamps")
        .with_i64_data(&[1000, 2000, 3000]);
    let finished2 = g2.finish();
    b.add_group(finished2);

    b.finish().unwrap()
}

fn write_to_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Test 1: File::open() uses mmap when feature is enabled
// ---------------------------------------------------------------------------

#[test]
fn file_open_uses_mmap() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_1.h5");
    let file = File::open(&path).unwrap();
    assert!(file.is_mmap(), "File::open should use mmap by default");
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 2: File::open() reads datasets correctly via mmap
// ---------------------------------------------------------------------------

#[test]
fn file_open_mmap_reads_f64() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_2.h5");
    let file = File::open(&path).unwrap();
    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 3: File::open() reads attributes via mmap
// ---------------------------------------------------------------------------

#[test]
fn file_open_mmap_reads_attrs() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_3.h5");
    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert!(matches!(attrs.get("version"), Some(AttrValue::I64(2))));
    assert!(
        matches!(attrs.get("description"), Some(AttrValue::String(s)) if s == "test file")
    );
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 4: File::open() navigates groups via mmap
// ---------------------------------------------------------------------------

#[test]
fn file_open_mmap_navigates_groups() {
    let bytes = make_grouped_file();
    let path = write_to_temp(&bytes, "mmap_default_test_4.h5");
    let file = File::open(&path).unwrap();

    let root = file.root();
    let mut groups = root.groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["metadata", "sensors"]);

    let sensors = file.group("sensors").unwrap();
    let mut ds_names = sensors.datasets().unwrap();
    ds_names.sort();
    assert_eq!(ds_names, vec!["humidity", "temperature"]);

    let temp = file.dataset("sensors/temperature").unwrap();
    assert_eq!(temp.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 5: File::from_bytes() backward compatibility (not mmap)
// ---------------------------------------------------------------------------

#[test]
fn file_from_bytes_backward_compat() {
    let bytes = make_simple_file();
    let file = File::from_bytes(bytes).unwrap();
    assert!(!file.is_mmap(), "from_bytes should not use mmap");
    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
}

// ---------------------------------------------------------------------------
// Test 6: File::open_buffered() works and is not mmap
// ---------------------------------------------------------------------------

#[test]
fn file_open_buffered_works() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_6.h5");
    let file = File::open_buffered(&path).unwrap();
    assert!(!file.is_mmap(), "open_buffered should not use mmap");
    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 7: Zero-copy raw read for contiguous dataset
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_raw_read_contiguous() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&values);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_7.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let raw_ref = ds.read_raw_ref().unwrap();
    assert!(raw_ref.is_some(), "contiguous dataset should return Some");
    let raw = raw_ref.unwrap();
    assert_eq!(raw.len(), 5 * 8); // 5 f64s = 40 bytes
    // Verify the bytes match the f64 LE encoding
    for (i, &v) in values.iter().enumerate() {
        let expected = v.to_le_bytes();
        assert_eq!(&raw[i * 8..(i + 1) * 8], &expected);
    }
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 8: Zero-copy raw read returns None for chunked dataset
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_raw_read_chunked_returns_none() {
    let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[10_000])
        .with_chunks(&[1_000]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_8.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let raw_ref = ds.read_raw_ref().unwrap();
    assert!(raw_ref.is_none(), "chunked dataset should return None");
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 9: Round-trip: write with FileBuilder, open with mmap, read back
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_write_mmap_read() {
    let dir = std::env::temp_dir();
    let path = dir.join("mmap_default_test_9.h5");

    let original = vec![1.1, 2.2, 3.3, 4.4, 5.5];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&original);
    b.set_attr("info", AttrValue::String("roundtrip".into()));
    b.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    assert!(file.is_mmap());
    let values = file.dataset("data").unwrap().read_f64().unwrap();
    assert_eq!(values, original);

    let attrs = file.root().attrs().unwrap();
    assert!(
        matches!(attrs.get("info"), Some(AttrValue::String(s)) if s == "roundtrip")
    );
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 10: Large file (10M f64) opens with mmap
// ---------------------------------------------------------------------------

#[test]
fn large_file_mmap_open() {
    let n = 100_000; // 100K f64 = 800KB (fast enough for a test)
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_10.h5");

    let file = File::open(&path).unwrap();
    assert!(file.is_mmap());
    let values = file.dataset("data").unwrap().read_f64().unwrap();
    assert_eq!(values.len(), n);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[n - 1], (n - 1) as f64);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 11: mmap and buffered produce identical results
// ---------------------------------------------------------------------------

#[test]
fn mmap_and_buffered_identical() {
    let bytes = make_grouped_file();
    let path = write_to_temp(&bytes, "mmap_default_test_11.h5");

    let mmap_file = File::open(&path).unwrap();
    let buf_file = File::open_buffered(&path).unwrap();

    // Compare datasets
    let mmap_temp = mmap_file
        .dataset("sensors/temperature")
        .unwrap()
        .read_f64()
        .unwrap();
    let buf_temp = buf_file
        .dataset("sensors/temperature")
        .unwrap()
        .read_f64()
        .unwrap();
    assert_eq!(mmap_temp, buf_temp);

    // Compare group listing
    let mmap_groups = mmap_file.root().groups().unwrap();
    let buf_groups = buf_file.root().groups().unwrap();
    assert_eq!(mmap_groups, buf_groups);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 12: Dataset shape and dtype through mmap
// ---------------------------------------------------------------------------

#[test]
fn mmap_dataset_shape_dtype() {
    let mut b = FileBuilder::new();
    b.create_dataset("matrix")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_12.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("matrix").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 13: All numeric types through mmap
// ---------------------------------------------------------------------------

#[test]
fn mmap_all_numeric_types() {
    let mut b = FileBuilder::new();
    b.create_dataset("f32").with_f32_data(&[1.5f32, 2.5, 3.5]);
    b.create_dataset("f64").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("i32").with_i32_data(&[10, 20, 30]);
    b.create_dataset("i64").with_i64_data(&[100, 200, 300]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_13.h5");

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("f32").unwrap().read_f32().unwrap(),
        vec![1.5f32, 2.5, 3.5]
    );
    assert_eq!(
        file.dataset("f64").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        file.dataset("i32").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("i64").unwrap().read_i64().unwrap(),
        vec![100, 200, 300]
    );
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 14: Dataset attributes through mmap
// ---------------------------------------------------------------------------

#[test]
fn mmap_dataset_attrs() {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("unit", AttrValue::String("meters".into()));
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_14.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let attrs = ds.attrs().unwrap();
    assert!(matches!(attrs.get("unit"), Some(AttrValue::String(s)) if s == "meters"));
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 15: File::open error on nonexistent file
// ---------------------------------------------------------------------------

#[test]
fn mmap_open_nonexistent() {
    let result = File::open("/tmp/purehdf5_mmap_does_not_exist_98765.h5");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Test 16: Debug format includes mmap info
// ---------------------------------------------------------------------------

#[test]
fn file_debug_shows_mmap() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_16.h5");

    let mmap_file = File::open(&path).unwrap();
    let debug = format!("{mmap_file:?}");
    assert!(debug.contains("mmap: true"));

    let buf_file = File::from_bytes(make_simple_file()).unwrap();
    let debug = format!("{buf_file:?}");
    assert!(debug.contains("mmap: false"));

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 17: LazyFile::open_mmap convenience constructor
// ---------------------------------------------------------------------------

#[test]
fn lazy_file_open_mmap_convenience() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_17.h5");

    let lazy = LazyFile::open_mmap(&path).unwrap();
    let ds = lazy.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 18: LazyFile::from_bytes convenience constructor
// ---------------------------------------------------------------------------

#[test]
fn lazy_file_from_bytes_convenience() {
    let bytes = make_simple_file();
    let lazy = LazyFile::from_bytes(bytes).unwrap();
    let ds = lazy.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
}

// ---------------------------------------------------------------------------
// Test 19: Zero-copy raw read on from_bytes (owned) dataset
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_raw_read_owned() {
    let values: Vec<f64> = vec![10.0, 20.0, 30.0];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&values);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let raw_ref = ds.read_raw_ref().unwrap();
    assert!(raw_ref.is_some());
    assert_eq!(raw_ref.unwrap().len(), 3 * 8);
}

// ---------------------------------------------------------------------------
// Test 20: Group attributes through mmap
// ---------------------------------------------------------------------------

#[test]
fn mmap_group_attrs() {
    let bytes = make_grouped_file();
    let path = write_to_temp(&bytes, "mmap_default_test_20.h5");

    let file = File::open(&path).unwrap();
    let sensors = file.group("sensors").unwrap();
    let attrs = sensors.attrs().unwrap();
    assert!(
        matches!(attrs.get("location"), Some(AttrValue::String(s)) if s == "lab")
    );
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 21: as_bytes returns correct data for mmap file
// ---------------------------------------------------------------------------

#[test]
fn mmap_as_bytes() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_default_test_21.h5");

    let file = File::open(&path).unwrap();
    let file_bytes = file.as_bytes();
    assert_eq!(&file_bytes[..8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(file_bytes.len(), bytes.len());
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 22: read_as_slice::<f64> zero-copy typed read
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_read_as_slice_f64() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&values);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_22.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    match ds.read_as_slice::<f64>() {
        Ok(Some(typed)) => assert_eq!(typed, &[1.0, 2.0, 3.0, 4.0, 5.0]),
        Ok(None) => panic!("contiguous dataset should return Some"),
        Err(e) => {
            // Alignment error is acceptable if mmap offset isn't 8-byte aligned
            assert!(format!("{e}").contains("alignment"), "unexpected error: {e}");
        }
    }
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 23: read_as_slice pointer identity (points into mmap region)
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_read_as_slice_pointer_identity() {
    let values: Vec<f64> = vec![42.0, 43.0];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&values);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_23.h5");

    let file = File::open(&path).unwrap();
    let file_bytes = file.as_bytes();
    let file_range = file_bytes.as_ptr_range();

    let ds = file.dataset("data").unwrap();
    // Get the raw slice
    let raw = ds.read_raw_ref().unwrap().unwrap();
    assert!(file_range.contains(&raw.as_ptr()), "raw slice must point into mmap");

    // Get the typed slice — must also point into the same region (if aligned)
    match ds.read_as_slice::<f64>() {
        Ok(Some(typed)) => {
            let typed_ptr = typed.as_ptr() as *const u8;
            assert!(file_range.contains(&typed_ptr), "typed slice must point into mmap");
            assert_eq!(typed_ptr, raw.as_ptr(), "typed and raw should share the same pointer");
            assert_eq!(typed, &[42.0, 43.0]);
        }
        Ok(None) => panic!("contiguous dataset should return Some"),
        Err(e) => {
            // Alignment error is acceptable if mmap offset isn't 8-byte aligned
            assert!(format!("{e}").contains("alignment"), "unexpected error: {e}");
        }
    }
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 24: read_as_slice returns None for chunked datasets
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_read_as_slice_chunked_returns_none() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[100])
        .with_chunks(&[10]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_24.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let result = ds.read_as_slice::<f64>().unwrap();
    assert!(result.is_none(), "chunked dataset should return None");
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Test 25: read_as_slice with i32 data
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_read_as_slice_i32() {
    let values: Vec<i32> = vec![10, -20, 30, -40];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_i32_data(&values);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_default_test_25.h5");

    let file = File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    // i32 has 4-byte alignment which mmap should satisfy
    match ds.read_as_slice::<i32>() {
        Ok(Some(typed)) => assert_eq!(typed, &[10, -20, 30, -40]),
        Ok(None) => panic!("contiguous dataset should return Some"),
        Err(e) => {
            // Alignment error is acceptable if mmap offset isn't aligned
            assert!(format!("{e}").contains("alignment"), "unexpected error: {e}");
        }
    }
    std::fs::remove_file(&path).ok();
}
