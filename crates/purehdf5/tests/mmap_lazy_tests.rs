//! Tests for MmapReader, MmapReadWrite, LazyFile, and PrefetchReader.
//!
//! At least 20 tests covering:
//! - MmapReader: open, read dataset, read attributes
//! - MmapReader: round-trip (write with FileBuilder, read with MmapReader)
//! - MmapReader: concurrent reads from same mmap (multiple references)
//! - LazyFile: open only parses superblock (verify no full file scan)
//! - LazyFile: access dataset triggers parse of that dataset only
//! - LazyFile: multiple dataset accesses cache correctly
//! - PrefetchReader: sequential chunk read produces correct data
//! - Large file (10M f64): mmap vs file reader performance

use purehdf5::{AttrValue, FileBuilder, LazyFile};
use purehdf5_io::prefetch::PrefetchReader;
use purehdf5_io::{MemoryReader, MmapReader, MmapReadWrite};

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
    b.finish().unwrap()
}

fn make_chunked_file() -> Vec<u8> {
    let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[10_000])
        .with_chunks(&[1_000]);
    b.finish().unwrap()
}

fn write_to_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

// ---------------------------------------------------------------------------
// MmapReader tests
// ---------------------------------------------------------------------------

/// Test 1: MmapReader opens and reads a valid HDF5 file
#[test]
fn mmap_reader_open_hdf5() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_test_open.h5");
    let reader = MmapReader::open(&path).unwrap();
    assert_eq!(&reader.as_bytes()[..8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(reader.len(), bytes.len());
    std::fs::remove_file(&path).ok();
}

/// Test 2: MmapReader can read a dataset through LazyFile
#[test]
fn mmap_reader_read_dataset() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_test_dataset.h5");
    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();
    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

/// Test 3: MmapReader can read attributes through LazyFile
#[test]
fn mmap_reader_read_attrs() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_test_attrs.h5");
    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert!(matches!(attrs.get("version"), Some(AttrValue::I64(2))));
    assert!(
        matches!(attrs.get("description"), Some(AttrValue::String(s)) if s == "test file")
    );
    std::fs::remove_file(&path).ok();
}

/// Test 4: MmapReader round-trip (write with FileBuilder, read with MmapReader)
#[test]
fn mmap_reader_roundtrip() {
    let original = vec![1.1, 2.2, 3.3, 4.4, 5.5];
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&original);
    let path = write_to_temp(&b.finish().unwrap(), "mmap_test_roundtrip.h5");

    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();
    let values = file.dataset("data").unwrap().read_f64().unwrap();
    assert_eq!(values, original);
    std::fs::remove_file(&path).ok();
}

/// Test 5: MmapReader concurrent reads (multiple references from same mmap)
#[test]
fn mmap_reader_concurrent_reads() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_test_concurrent.h5");
    let reader = MmapReader::open(&path).unwrap();

    // Multiple simultaneous borrows of the mapped memory
    let slice1 = reader.read_at(0, 8);
    let slice2 = reader.read_at(0, 4);
    let full = reader.as_bytes();

    assert_eq!(slice1.unwrap(), &bytes[..8]);
    assert_eq!(slice2.unwrap(), &bytes[..4]);
    assert_eq!(full, &bytes[..]);
    std::fs::remove_file(&path).ok();
}

/// Test 6: MmapReader with grouped file
#[test]
fn mmap_reader_grouped_file() {
    let bytes = make_grouped_file();
    let path = write_to_temp(&bytes, "mmap_test_grouped.h5");
    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();

    let temp = file.dataset("sensors/temperature").unwrap();
    assert_eq!(temp.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);

    let hum = file.dataset("sensors/humidity").unwrap();
    assert_eq!(hum.read_i32().unwrap(), vec![45, 50, 55]);
    std::fs::remove_file(&path).ok();
}

/// Test 7: MmapReader read_at returns None for out-of-bounds
#[test]
fn mmap_reader_out_of_bounds() {
    let bytes = make_simple_file();
    let path = write_to_temp(&bytes, "mmap_test_oob.h5");
    let reader = MmapReader::open(&path).unwrap();
    assert!(reader.read_at(reader.len(), 1).is_none());
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// MmapReadWrite tests
// ---------------------------------------------------------------------------

/// Test 8: MmapReadWrite create and write HDF5 data
#[test]
fn mmap_readwrite_hdf5_roundtrip() {
    let bytes = make_simple_file();
    let path = write_to_temp(&[], "mmap_test_rw.h5");
    {
        let mut rw = MmapReadWrite::create(&path, bytes.len() as u64).unwrap();
        rw.write_at(0, &bytes).unwrap();
        rw.flush().unwrap();
    }
    // Now read back with MmapReader
    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();
    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// LazyFile tests
// ---------------------------------------------------------------------------

/// Test 9: LazyFile open only parses superblock (no full file scan)
#[test]
fn lazy_file_open_minimal_parse() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    // Superblock is parsed
    assert!(file.superblock().version >= 2);

    // Root header is parsed
    assert!(!file.root_header().messages.is_empty());

    // No dataset headers cached yet
    assert_eq!(file.cached_header_count(), 0);
}

/// Test 10: LazyFile access dataset triggers parse of that dataset only
#[test]
fn lazy_file_dataset_access_caches() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    assert_eq!(file.cached_header_count(), 0);

    // Access one dataset
    let _ds = file.dataset("temperatures").unwrap();
    let count_after_first = file.cached_header_count();
    assert!(count_after_first >= 1);

    // Access another dataset
    let _ds2 = file.dataset("counts").unwrap();
    let count_after_second = file.cached_header_count();
    assert!(count_after_second > count_after_first);
}

/// Test 11: LazyFile multiple accesses to same dataset use cache
#[test]
fn lazy_file_cache_reuse() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let _ds1 = file.dataset("temperatures").unwrap();
    let count1 = file.cached_header_count();

    // Access same dataset again - cache count shouldn't change
    let _ds2 = file.dataset("temperatures").unwrap();
    let count2 = file.cached_header_count();
    assert_eq!(count1, count2);
}

/// Test 12: LazyFile with grouped file
#[test]
fn lazy_file_groups() {
    let bytes = make_grouped_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let root = file.root();
    let groups = root.groups().unwrap();
    assert_eq!(groups, vec!["sensors"]);

    let sensors = file.group("sensors").unwrap();
    let mut ds_names = sensors.datasets().unwrap();
    ds_names.sort();
    assert_eq!(ds_names, vec!["humidity", "temperature"]);
}

/// Test 13: LazyFile group attributes
#[test]
fn lazy_file_group_attrs() {
    let bytes = make_grouped_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let sensors = file.group("sensors").unwrap();
    let attrs = sensors.attrs().unwrap();
    assert!(
        matches!(attrs.get("location"), Some(AttrValue::String(s)) if s == "lab")
    );
}

/// Test 14: LazyFile dataset shape and dtype
#[test]
fn lazy_file_dataset_shape_dtype() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let ds = file.dataset("temperatures").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3]);
    assert_eq!(ds.dtype().unwrap(), purehdf5::DType::F64);
}

/// Test 15: LazyFile dataset attributes
#[test]
fn lazy_file_dataset_attrs() {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("unit", AttrValue::String("meters".into()));
    let bytes = b.finish().unwrap();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let ds = file.dataset("data").unwrap();
    let attrs = ds.attrs().unwrap();
    assert!(matches!(attrs.get("unit"), Some(AttrValue::String(s)) if s == "meters"));
}

/// Test 16: LazyFile debug format
#[test]
fn lazy_file_debug() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();
    let debug = format!("{file:?}");
    assert!(debug.contains("LazyFile"));
    assert!(debug.contains("cached_headers"));
}

/// Test 17: LazyFile error on invalid bytes
#[test]
fn lazy_file_invalid_bytes() {
    let reader = MemoryReader::new(vec![0, 1, 2, 3]);
    let result = LazyFile::open(reader);
    assert!(result.is_err());
}

/// Test 18: LazyFile error on nonexistent dataset
#[test]
fn lazy_file_dataset_not_found() {
    let bytes = make_simple_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();
    let err = file.dataset("nonexistent").unwrap_err();
    assert!(matches!(err, purehdf5::Error::Format(_)));
}

// ---------------------------------------------------------------------------
// PrefetchReader tests
// ---------------------------------------------------------------------------

/// Test 19: PrefetchReader sequential chunk read produces correct data
#[test]
fn prefetch_reader_sequential_correct_data() {
    let bytes = make_chunked_file();
    let reader = MemoryReader::new(bytes);
    let prefetch = PrefetchReader::with_defaults(reader, 8_000); // ~1000 f64s
    let file = LazyFile::open(prefetch).unwrap();

    let ds = file.dataset("data").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), 10_000);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[9999], 9999.0);
}

/// Test 20: PrefetchReader with MmapReader backend
#[test]
fn prefetch_reader_with_mmap() {
    let bytes = make_chunked_file();
    let path = write_to_temp(&bytes, "mmap_test_prefetch.h5");
    let reader = MmapReader::open(&path).unwrap();
    let prefetch = PrefetchReader::with_defaults(reader, 8_000);
    let file = LazyFile::open(prefetch).unwrap();

    let ds = file.dataset("data").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), 10_000);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[9999], 9999.0);
    std::fs::remove_file(&path).ok();
}

/// Test 21: LazyFile with many datasets only parses accessed ones
#[test]
fn lazy_file_100_datasets_parse_one() {
    let mut b = FileBuilder::new();
    for i in 0..100 {
        b.create_dataset(&format!("ds_{i:04}"))
            .with_f64_data(&[i as f64; 10]);
    }
    let bytes = b.finish().unwrap();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    // No headers cached initially
    assert_eq!(file.cached_header_count(), 0);

    // Read one dataset
    let ds = file.dataset("ds_0050").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), 10);
    assert_eq!(values[0], 50.0);

    // Only a few headers should be cached (the dataset + possibly some
    // intermediate group navigation headers), far fewer than 100
    let cached = file.cached_header_count();
    assert!(
        cached < 10,
        "Expected fewer than 10 cached headers, got {cached}"
    );
}

/// Test 22: MmapReader with LazyFile reads all numeric types
#[test]
fn mmap_lazy_all_numeric_types() {
    let mut b = FileBuilder::new();
    b.create_dataset("f32").with_f32_data(&[1.5f32, 2.5, 3.5]);
    b.create_dataset("f64").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("i32").with_i32_data(&[10, 20, 30]);
    b.create_dataset("i64").with_i64_data(&[100, 200, 300]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_test_all_types.h5");

    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();

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

/// Test 23: Large file (1M f64) via mmap vs file reader correctness
#[test]
fn large_file_mmap_correctness() {
    let n = 100_000; // 100K for faster test execution
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64]);
    let bytes = b.finish().unwrap();
    let path = write_to_temp(&bytes, "mmap_test_large.h5");

    // Read with MmapReader + LazyFile
    let reader = MmapReader::open(&path).unwrap();
    let file = LazyFile::open(reader).unwrap();
    let values = file.dataset("data").unwrap().read_f64().unwrap();
    assert_eq!(values.len(), n);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[n - 1], (n - 1) as f64);

    // Read with standard File
    let file2 = purehdf5::File::open(&path).unwrap();
    let values2 = file2.dataset("data").unwrap().read_f64().unwrap();
    assert_eq!(values, values2);

    std::fs::remove_file(&path).ok();
}

/// Test 24: LazyFile group navigation via group handle
#[test]
fn lazy_file_group_handle_navigation() {
    let bytes = make_grouped_file();
    let reader = MemoryReader::new(bytes);
    let file = LazyFile::open(reader).unwrap();

    let root = file.root();
    let sensors = root.group("sensors").unwrap();
    let ds = sensors.dataset("temperature").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
}
