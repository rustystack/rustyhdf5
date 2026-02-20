//! High-level API for reading and writing HDF5 files.
//!
//! This crate provides an ergonomic interface on top of `purehdf5-format`.
//!
//! # Reading
//!
//! ```no_run
//! use purehdf5::File;
//!
//! let file = File::open("data.h5").unwrap();
//! let ds = file.dataset("sensors/temperature").unwrap();
//! let values = ds.read_f64().unwrap();
//! println!("shape: {:?}, data: {:?}", ds.shape().unwrap(), values);
//! ```
//!
//! # Writing
//!
//! ```no_run
//! use purehdf5::{FileBuilder, AttrValue};
//!
//! let mut builder = FileBuilder::new();
//! builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
//! builder.set_attr("version", AttrValue::I64(1));
//! builder.write("output.h5").unwrap();
//! ```

pub mod error;
pub mod lazy;
#[cfg(feature = "mmap")]
pub mod mmap_file;
pub mod reader;
pub mod types;
pub mod writer;

pub use error::Error;
pub use lazy::{LazyDataset, LazyFile, LazyGroup};
#[cfg(feature = "mmap")]
pub use mmap_file::{MmapDataset, MmapFile, MmapGroup};
pub use reader::{Dataset, File, Group};
pub use types::{AttrValue, DType};
pub use writer::FileBuilder;

// Re-export useful types from purehdf5-format for advanced users
pub use purehdf5_format::type_builders::{CompoundTypeBuilder, EnumTypeBuilder};

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: create a simple HDF5 file in memory via FileBuilder
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Reading tests
    // -----------------------------------------------------------------------

    #[test]
    fn open_from_bytes() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        assert!(file.superblock().version >= 2);
    }

    #[test]
    fn read_f64_dataset() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("temperatures").unwrap();
        let values = ds.read_f64().unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn read_i32_dataset() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("counts").unwrap();
        let values = ds.read_i32().unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn dataset_shape() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("temperatures").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![3]);
    }

    #[test]
    fn dataset_dtype_f64() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("temperatures").unwrap();
        assert_eq!(ds.dtype().unwrap(), DType::F64);
    }

    #[test]
    fn dataset_dtype_i32() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("counts").unwrap();
        assert_eq!(ds.dtype().unwrap(), DType::I32);
    }

    #[test]
    fn root_group_datasets() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let root = file.root();
        let mut names = root.datasets().unwrap();
        names.sort();
        assert_eq!(names, vec!["counts", "temperatures"]);
    }

    #[test]
    fn root_group_attrs() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let root = file.root();
        let attrs = root.attrs().unwrap();
        assert!(matches!(attrs.get("version"), Some(AttrValue::I64(2))));
        assert!(matches!(attrs.get("description"), Some(AttrValue::String(s)) if s == "test file"));
    }

    #[test]
    fn navigate_groups() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let root = file.root();
        let mut group_names = root.groups().unwrap();
        group_names.sort();
        assert_eq!(group_names, vec!["metadata", "sensors"]);
    }

    #[test]
    fn read_nested_dataset() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let ds = file.dataset("sensors/temperature").unwrap();
        let values = ds.read_f64().unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn read_nested_i32_dataset() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let ds = file.dataset("sensors/humidity").unwrap();
        let values = ds.read_i32().unwrap();
        assert_eq!(values, vec![45, 50, 55]);
    }

    #[test]
    fn group_handle_datasets() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let sensors = file.group("sensors").unwrap();
        let mut names = sensors.datasets().unwrap();
        names.sort();
        assert_eq!(names, vec!["humidity", "temperature"]);
    }

    #[test]
    fn group_handle_attrs() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let sensors = file.group("sensors").unwrap();
        let attrs = sensors.attrs().unwrap();
        assert!(matches!(attrs.get("location"), Some(AttrValue::String(s)) if s == "lab"));
    }

    #[test]
    fn group_get_dataset() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let sensors = file.group("sensors").unwrap();
        let ds = sensors.dataset("temperature").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn group_get_subgroup() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        let root = file.root();
        let sensors = root.group("sensors").unwrap();
        let names = sensors.datasets().unwrap();
        assert_eq!(names.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Error case tests
    // -----------------------------------------------------------------------

    #[test]
    fn dataset_not_found() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let err = file.dataset("nonexistent").unwrap_err();
        assert!(matches!(err, Error::Format(_)));
    }

    #[test]
    fn not_a_dataset_error() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();
        let err = file.dataset("sensors").unwrap_err();
        assert!(matches!(err, Error::NotADataset(_)));
    }

    #[test]
    fn open_invalid_bytes() {
        let err = File::from_bytes(vec![0, 1, 2, 3]).unwrap_err();
        assert!(matches!(err, Error::Format(_)));
    }

    #[test]
    fn open_empty_bytes() {
        let err = File::from_bytes(Vec::new()).unwrap_err();
        assert!(matches!(err, Error::Format(_)));
    }

    // -----------------------------------------------------------------------
    // Writing tests
    // -----------------------------------------------------------------------

    #[test]
    fn file_builder_simple() {
        let mut b = FileBuilder::new();
        b.create_dataset("x").with_f64_data(&[1.0, 2.0]);
        let bytes = b.finish().unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
    }

    #[test]
    fn file_builder_with_group() {
        let mut b = FileBuilder::new();
        let mut g = b.create_group("grp");
        g.create_dataset("vals").with_i32_data(&[1, 2, 3]);
        let finished = g.finish();
        b.add_group(finished);
        let bytes = b.finish().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn file_builder_with_attrs() {
        let mut b = FileBuilder::new();
        b.set_attr("name", AttrValue::String("test".into()));
        b.set_attr("count", AttrValue::I64(42));
        b.create_dataset("d").with_f64_data(&[2.78]);
        let bytes = b.finish().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn file_builder_dataset_with_shape() {
        let mut b = FileBuilder::new();
        b.create_dataset("matrix")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[2, 3]);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("matrix").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![2, 3]);
        assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn file_builder_dataset_with_attrs() {
        let mut b = FileBuilder::new();
        b.create_dataset("data")
            .with_f64_data(&[1.0])
            .set_attr("unit", AttrValue::String("meters".into()));
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("data").unwrap();
        let attrs = ds.attrs().unwrap();
        assert!(matches!(attrs.get("unit"), Some(AttrValue::String(s)) if s == "meters"));
    }

    // -----------------------------------------------------------------------
    // Round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_f64() {
        let original = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let mut b = FileBuilder::new();
        b.create_dataset("data").with_f64_data(&original);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let values = file.dataset("data").unwrap().read_f64().unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn roundtrip_i32() {
        let original = vec![-10, 0, 10, 100, -100];
        let mut b = FileBuilder::new();
        b.create_dataset("data").with_i32_data(&original);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let values = file.dataset("data").unwrap().read_i32().unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn roundtrip_i64() {
        let original = vec![i64::MIN, -1, 0, 1, i64::MAX];
        let mut b = FileBuilder::new();
        b.create_dataset("data").with_i64_data(&original);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let values = file.dataset("data").unwrap().read_i64().unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn roundtrip_u8() {
        let original = vec![0u8, 127, 255];
        let mut b = FileBuilder::new();
        b.create_dataset("data").with_u8_data(&original);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.dtype().unwrap(), DType::U8);
    }

    #[test]
    fn roundtrip_f32() {
        let original = vec![1.5f32, 2.5, 3.5];
        let mut b = FileBuilder::new();
        b.create_dataset("data").with_f32_data(&original);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.dtype().unwrap(), DType::F32);
        let values = ds.read_f32().unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn roundtrip_attrs() {
        let mut b = FileBuilder::new();
        b.set_attr("f64_val", AttrValue::F64(2.78));
        b.set_attr("i64_val", AttrValue::I64(-42));
        b.set_attr("str_val", AttrValue::String("hello".into()));
        b.set_attr(
            "f64_arr",
            AttrValue::F64Array(vec![1.0, 2.0, 3.0]),
        );
        b.create_dataset("d").with_f64_data(&[0.0]);
        let bytes = b.finish().unwrap();

        let file = File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();

        assert!(matches!(attrs.get("f64_val"), Some(AttrValue::F64(v)) if (*v - 2.78).abs() < 1e-10));
        assert!(matches!(attrs.get("i64_val"), Some(AttrValue::I64(-42))));
        assert!(matches!(attrs.get("str_val"), Some(AttrValue::String(s)) if s == "hello"));
        assert!(matches!(attrs.get("f64_arr"), Some(AttrValue::F64Array(arr)) if arr == &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn roundtrip_grouped() {
        let bytes = make_grouped_file();
        let file = File::from_bytes(bytes).unwrap();

        // Check groups
        let root = file.root();
        let mut groups = root.groups().unwrap();
        groups.sort();
        assert_eq!(groups, vec!["metadata", "sensors"]);

        // Check sensors group
        let sensors = file.group("sensors").unwrap();
        let mut ds_names = sensors.datasets().unwrap();
        ds_names.sort();
        assert_eq!(ds_names, vec!["humidity", "temperature"]);

        // Read data
        let temp = file.dataset("sensors/temperature").unwrap();
        assert_eq!(temp.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);

        let hum = file.dataset("sensors/humidity").unwrap();
        assert_eq!(hum.read_i32().unwrap(), vec![45, 50, 55]);

        // Check metadata group
        let ts = file.dataset("metadata/timestamps").unwrap();
        assert_eq!(ts.read_i64().unwrap(), vec![1000, 2000, 3000]);
    }

    #[test]
    fn file_builder_write_to_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_test_write.h5");

        let mut b = FileBuilder::new();
        b.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);
        b.write(&path).unwrap();

        let file = File::open(&path).unwrap();
        let values = file.dataset("x").unwrap().read_f64().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_debug_impl() {
        let bytes = make_simple_file();
        let file = File::from_bytes(bytes).unwrap();
        let debug = format!("{file:?}");
        assert!(debug.contains("File"));
        assert!(debug.contains("size"));
    }

    #[test]
    fn dtype_display() {
        assert_eq!(DType::F64.to_string(), "f64");
        assert_eq!(DType::I32.to_string(), "i32");
        assert_eq!(DType::String.to_string(), "string");
        assert_eq!(
            DType::Compound(vec![
                ("x".into(), DType::F64),
                ("y".into(), DType::F64),
            ])
            .to_string(),
            "compound{x: f64, y: f64}"
        );
    }

    #[test]
    fn error_display() {
        let err = Error::NotADataset("foo".into());
        assert_eq!(err.to_string(), "not a dataset: foo");

        let err = Error::Format(purehdf5_format::error::FormatError::SignatureNotFound);
        assert!(err.to_string().contains("format error"));
    }
}
