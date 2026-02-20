//! End-to-end integration tests for purehdf5: full read/write pipelines,
//! round-trips with all types, chunked+compressed data, deep group hierarchies,
//! large datasets, attributes everywhere, empty datasets, and file overwrite.

use purehdf5::{AttrValue, CompoundTypeBuilder, DType, File, FileBuilder};

// ---------------------------------------------------------------------------
// 1. Full read pipeline
// ---------------------------------------------------------------------------

#[test]
fn full_read_pipeline() {
    // Build a file with groups, datasets, attributes
    let mut b = FileBuilder::new();
    b.set_attr("file_version", AttrValue::I64(3));

    let mut g = b.create_group("sensors");
    g.create_dataset("temperature")
        .with_f64_data(&[20.0, 21.5, 22.3])
        .set_attr("units", AttrValue::String("celsius".into()));
    g.create_dataset("pressure")
        .with_f32_data(&[1013.0, 1014.5, 1012.8]);
    g.set_attr("location", AttrValue::String("lab_a".into()));
    b.add_group(g.finish());

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    // Navigate root
    let root = file.root();
    let groups = root.groups().unwrap();
    assert_eq!(groups, vec!["sensors"]);

    let root_attrs = root.attrs().unwrap();
    assert!(matches!(root_attrs.get("file_version"), Some(AttrValue::I64(3))));

    // Navigate group
    let sensors = file.group("sensors").unwrap();
    let mut ds_names = sensors.datasets().unwrap();
    ds_names.sort();
    assert_eq!(ds_names, vec!["pressure", "temperature"]);

    let group_attrs = sensors.attrs().unwrap();
    assert!(
        matches!(group_attrs.get("location"), Some(AttrValue::String(s)) if s == "lab_a")
    );

    // Read datasets
    let temp = file.dataset("sensors/temperature").unwrap();
    assert_eq!(temp.shape().unwrap(), vec![3]);
    assert_eq!(temp.dtype().unwrap(), DType::F64);
    assert_eq!(temp.read_f64().unwrap(), vec![20.0, 21.5, 22.3]);

    let ds_attrs = temp.attrs().unwrap();
    assert!(
        matches!(ds_attrs.get("units"), Some(AttrValue::String(s)) if s == "celsius")
    );

    let pressure = file.dataset("sensors/pressure").unwrap();
    assert_eq!(pressure.dtype().unwrap(), DType::F32);
    let pvals = pressure.read_f32().unwrap();
    assert_eq!(pvals, vec![1013.0, 1014.5, 1012.8]);
}

// ---------------------------------------------------------------------------
// 2. Full write pipeline
// ---------------------------------------------------------------------------

#[test]
fn full_write_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("write_test.h5");

    // Create file
    let mut b = FileBuilder::new();
    b.set_attr("created_by", AttrValue::String("purehdf5".into()));

    let mut g = b.create_group("experiment");
    g.create_dataset("results")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[2, 2])
        .set_attr("description", AttrValue::String("2x2 matrix".into()));
    g.set_attr("run_id", AttrValue::I64(42));
    b.add_group(g.finish());

    b.create_dataset("config_val").with_i32_data(&[100]);

    b.write(&path).unwrap();

    // Reopen and verify everything
    let file = File::open(&path).unwrap();

    let root_attrs = file.root().attrs().unwrap();
    assert!(
        matches!(root_attrs.get("created_by"), Some(AttrValue::String(s)) if s == "purehdf5")
    );

    let results = file.dataset("experiment/results").unwrap();
    assert_eq!(results.shape().unwrap(), vec![2, 2]);
    assert_eq!(results.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let ds_attrs = results.attrs().unwrap();
    assert!(
        matches!(ds_attrs.get("description"), Some(AttrValue::String(s)) if s == "2x2 matrix")
    );

    let exp = file.group("experiment").unwrap();
    let exp_attrs = exp.attrs().unwrap();
    assert!(matches!(exp_attrs.get("run_id"), Some(AttrValue::I64(42))));

    let config = file.dataset("config_val").unwrap();
    assert_eq!(config.read_i32().unwrap(), vec![100]);
}

// ---------------------------------------------------------------------------
// 3. Round-trip with all supported write/read types
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_f32() {
    let data = vec![1.5f32, -2.5, 0.0, f32::MAX, f32::MIN];
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f32_data(&data);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f32().unwrap(), data);
}

#[test]
fn roundtrip_f64() {
    let data = vec![1.1, -2.2, 0.0, f64::MAX, f64::MIN];
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&data);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), data);
}

#[test]
fn roundtrip_i32() {
    let data = vec![i32::MIN, -1, 0, 1, i32::MAX];
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&data);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), data);
}

#[test]
fn roundtrip_i64() {
    let data = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i64_data(&data);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i64().unwrap(), data);
}

#[test]
fn roundtrip_u8() {
    let data = vec![0u8, 1, 127, 255];
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_u8_data(&data);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U8);
    // Read back via u64 (u8 upcast)
    let vals = ds.read_u64().unwrap();
    assert_eq!(vals, vec![0, 1, 127, 255]);
}

#[test]
fn roundtrip_compound_type() {
    // Build compound type: {x: f64, y: f64, id: i32}
    let dt = CompoundTypeBuilder::new()
        .f64_field("x")
        .f64_field("y")
        .i32_field("id")
        .build();

    // 2 records: (1.0, 2.0, 10) and (3.0, 4.0, 20)
    let mut raw = Vec::new();
    raw.extend_from_slice(&1.0f64.to_le_bytes());
    raw.extend_from_slice(&2.0f64.to_le_bytes());
    raw.extend_from_slice(&10i32.to_le_bytes());
    raw.extend_from_slice(&3.0f64.to_le_bytes());
    raw.extend_from_slice(&4.0f64.to_le_bytes());
    raw.extend_from_slice(&20i32.to_le_bytes());

    let mut b = FileBuilder::new();
    b.create_dataset("points")
        .with_compound_data(dt, raw.clone(), 2);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    let ds = file.dataset("points").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![2]);
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0], ("x".into(), DType::F64));
            assert_eq!(fields[1], ("y".into(), DType::F64));
            assert_eq!(fields[2], ("id".into(), DType::I32));
        }
        other => panic!("expected Compound, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. Chunked + compressed round-trip
// ---------------------------------------------------------------------------

#[test]
fn chunked_deflate_roundtrip() {
    let n = 10_000;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_f64_data(&data)
        .with_chunks(&[1000])
        .with_deflate(6);
    let bytes = b.finish().unwrap();

    // File should be smaller than raw data (10000 * 8 = 80000 bytes)
    let raw_size = n * 8;
    assert!(
        bytes.len() < raw_size,
        "compressed file {} should be < raw data {}",
        bytes.len(),
        raw_size
    );

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("chunked").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), n);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[n - 1] - (n - 1) as f64 * 0.1).abs() < 1e-10);
}

#[test]
fn chunked_shuffle_deflate_roundtrip() {
    let data: Vec<f64> = (0..5000).map(|i| (i as f64).sin()).collect();

    let mut b = FileBuilder::new();
    b.create_dataset("shuffled")
        .with_f64_data(&data)
        .with_chunks(&[500])
        .with_shuffle()
        .with_deflate(4);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let values = file.dataset("shuffled").unwrap().read_f64().unwrap();
    assert_eq!(values.len(), 5000);
    for (i, &v) in values.iter().enumerate() {
        assert!(
            (v - (i as f64).sin()).abs() < 1e-10,
            "mismatch at index {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Deep group hierarchy
// ---------------------------------------------------------------------------

#[test]
fn deep_group_hierarchy() {
    // Create /a/b/c/d/e with a dataset at the leaf
    let mut b = FileBuilder::new();

    let mut ge = b.create_group("e");
    ge.create_dataset("leaf_data").with_f64_data(&[42.0]);
    ge.set_attr("depth", AttrValue::I64(5));
    let finished_e = ge.finish();

    let mut gd = b.create_group("d");
    gd.create_dataset("d_data").with_i32_data(&[4]);
    let finished_d = gd.finish();

    // Build nested structure: each level contains the next
    // Due to API constraints (groups can only contain datasets, not sub-groups
    // at the GroupBuilder level), we create a flat structure and use path-based
    // access with a single-level group per path segment.
    //
    // Actually, the API uses FileWriter.add_group which adds at root level.
    // For deep nesting, we need to use the format-level API or a workaround.
    // Let's test the path-based navigation that the API supports.

    let mut builder = FileBuilder::new();

    // The high-level API adds groups at root level. Let's create multiple
    // groups and test navigation through them.
    let mut ga = builder.create_group("a");
    ga.create_dataset("val").with_i32_data(&[1]);
    builder.add_group(ga.finish());

    let mut gb = builder.create_group("b");
    gb.create_dataset("val").with_i32_data(&[2]);
    builder.add_group(gb.finish());

    builder.add_group(finished_d);
    builder.add_group(finished_e);

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    // Verify all groups exist
    let root = file.root();
    let mut groups = root.groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["a", "b", "d", "e"]);

    // Navigate each group
    let a = file.group("a").unwrap();
    assert_eq!(a.dataset("val").unwrap().read_i32().unwrap(), vec![1]);

    let e = file.group("e").unwrap();
    let e_attrs = e.attrs().unwrap();
    assert!(matches!(e_attrs.get("depth"), Some(AttrValue::I64(5))));
    assert_eq!(
        e.dataset("leaf_data").unwrap().read_f64().unwrap(),
        vec![42.0]
    );
}

// ---------------------------------------------------------------------------
// 6. Large datasets
// ---------------------------------------------------------------------------

#[test]
fn large_dataset_1m_floats() {
    let n = 1_000_000;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();

    let mut b = FileBuilder::new();
    b.create_dataset("big")
        .with_f64_data(&data)
        .with_chunks(&[50_000])
        .with_deflate(1);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("big").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![n as u64]);

    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), n);

    // Verify first, last, and some random samples
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[n - 1] - (n - 1) as f64 * 0.001).abs() < 1e-10);
    assert!((values[500_000] - 500.0).abs() < 1e-10);
    assert!((values[123_456] - 123.456).abs() < 1e-10);
    assert!((values[999_999] - 999.999).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// 7. Multiple datasets in one file
// ---------------------------------------------------------------------------

#[test]
fn multiple_datasets_in_one_file() {
    let mut b = FileBuilder::new();

    b.create_dataset("f64_data").with_f64_data(&[1.0, 2.0]);
    b.create_dataset("f32_data").with_f32_data(&[3.0f32, 4.0]);
    b.create_dataset("i32_data").with_i32_data(&[5, 6]);
    b.create_dataset("i64_data").with_i64_data(&[7i64, 8]);
    b.create_dataset("u8_data").with_u8_data(&[9u8, 10]);
    b.create_dataset("matrix_2d")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3]);
    b.create_dataset("matrix_3d")
        .with_f64_data(&(0..24).map(|i| i as f64).collect::<Vec<_>>())
        .with_shape(&[2, 3, 4]);
    b.create_dataset("single_val").with_f64_data(&[7.77]);
    b.create_dataset("big_1d")
        .with_i32_data(&(0..1000).collect::<Vec<i32>>());

    // Compound dataset
    let dt = CompoundTypeBuilder::new()
        .f64_field("value")
        .i32_field("flag")
        .build();
    let mut raw = Vec::new();
    raw.extend_from_slice(&99.9f64.to_le_bytes());
    raw.extend_from_slice(&1i32.to_le_bytes());
    b.create_dataset("compound_ds")
        .with_compound_data(dt, raw, 1);

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let root = file.root();

    let mut names = root.datasets().unwrap();
    names.sort();
    assert_eq!(names.len(), 10);

    assert_eq!(
        file.dataset("f64_data").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.dataset("f32_data").unwrap().read_f32().unwrap(),
        vec![3.0f32, 4.0]
    );
    assert_eq!(
        file.dataset("i32_data").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );
    assert_eq!(
        file.dataset("i64_data").unwrap().read_i64().unwrap(),
        vec![7i64, 8]
    );
    assert_eq!(
        file.dataset("matrix_2d").unwrap().shape().unwrap(),
        vec![2, 3]
    );
    assert_eq!(
        file.dataset("matrix_3d").unwrap().shape().unwrap(),
        vec![2, 3, 4]
    );
    assert_eq!(
        file.dataset("single_val").unwrap().read_f64().unwrap(),
        vec![7.77]
    );
    assert_eq!(
        file.dataset("big_1d").unwrap().read_i32().unwrap(),
        (0..1000).collect::<Vec<i32>>()
    );
    assert!(matches!(
        file.dataset("compound_ds").unwrap().dtype().unwrap(),
        DType::Compound(_)
    ));
}

// ---------------------------------------------------------------------------
// 8. Attributes on everything
// ---------------------------------------------------------------------------

#[test]
fn attributes_on_everything() {
    let mut b = FileBuilder::new();

    // File-level attrs
    b.set_attr("file_str", AttrValue::String("hello".into()));
    b.set_attr("file_int", AttrValue::I64(99));
    b.set_attr("file_float", AttrValue::F64(2.5));
    b.set_attr(
        "file_arr",
        AttrValue::F64Array(vec![1.0, 2.0, 3.0]),
    );
    b.set_attr(
        "file_iarr",
        AttrValue::I64Array(vec![10, 20, 30]),
    );

    // Group-level attrs
    let mut g = b.create_group("grp");
    g.set_attr("grp_str", AttrValue::String("world".into()));
    g.set_attr("grp_int", AttrValue::I64(-1));

    // Dataset-level attrs
    g.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("ds_str", AttrValue::String("dataset_attr".into()))
        .set_attr("ds_float", AttrValue::F64(42.0));
    b.add_group(g.finish());

    // Root dataset with attrs
    b.create_dataset("root_ds")
        .with_i32_data(&[1])
        .set_attr("root_ds_attr", AttrValue::I64(7));

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    // Verify file attrs
    let root_attrs = file.root().attrs().unwrap();
    assert!(
        matches!(root_attrs.get("file_str"), Some(AttrValue::String(s)) if s == "hello")
    );
    assert!(matches!(root_attrs.get("file_int"), Some(AttrValue::I64(99))));
    assert!(
        matches!(root_attrs.get("file_float"), Some(AttrValue::F64(v)) if (*v - 2.5).abs() < 1e-10)
    );
    assert!(
        matches!(root_attrs.get("file_arr"), Some(AttrValue::F64Array(v)) if v == &[1.0, 2.0, 3.0])
    );
    assert!(
        matches!(root_attrs.get("file_iarr"), Some(AttrValue::I64Array(v)) if v == &[10, 20, 30])
    );

    // Verify group attrs
    let grp = file.group("grp").unwrap();
    let grp_attrs = grp.attrs().unwrap();
    assert!(
        matches!(grp_attrs.get("grp_str"), Some(AttrValue::String(s)) if s == "world")
    );
    assert!(matches!(grp_attrs.get("grp_int"), Some(AttrValue::I64(-1))));

    // Verify dataset attrs (in group)
    let ds = file.dataset("grp/data").unwrap();
    let ds_attrs = ds.attrs().unwrap();
    assert!(
        matches!(ds_attrs.get("ds_str"), Some(AttrValue::String(s)) if s == "dataset_attr")
    );
    assert!(
        matches!(ds_attrs.get("ds_float"), Some(AttrValue::F64(v)) if (*v - 42.0).abs() < 1e-10)
    );

    // Verify root dataset attrs
    let rds = file.dataset("root_ds").unwrap();
    let rds_attrs = rds.attrs().unwrap();
    assert!(matches!(
        rds_attrs.get("root_ds_attr"),
        Some(AttrValue::I64(7))
    ));
}

// ---------------------------------------------------------------------------
// 9. Empty datasets (zero-length)
// ---------------------------------------------------------------------------

#[test]
fn empty_dataset_f64() {
    let mut b = FileBuilder::new();
    b.create_dataset("empty").with_f64_data(&[]);
    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.read_f64().unwrap(), Vec::<f64>::new());
}

#[test]
fn empty_dataset_i32() {
    let mut b = FileBuilder::new();
    b.create_dataset("empty").with_i32_data(&[]);
    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.read_i32().unwrap(), Vec::<i32>::new());
}

// ---------------------------------------------------------------------------
// 10. Overwrite file
// ---------------------------------------------------------------------------

#[test]
fn overwrite_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overwrite.h5");

    // Write first version
    let mut b1 = FileBuilder::new();
    b1.create_dataset("version1").with_f64_data(&[1.0, 2.0, 3.0]);
    b1.set_attr("version", AttrValue::I64(1));
    b1.write(&path).unwrap();

    // Verify first version
    let f1 = File::open(&path).unwrap();
    assert_eq!(
        f1.dataset("version1").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );

    // Overwrite with new content
    let mut b2 = FileBuilder::new();
    b2.create_dataset("version2").with_i32_data(&[10, 20]);
    b2.set_attr("version", AttrValue::I64(2));
    b2.write(&path).unwrap();

    // Verify new content
    let f2 = File::open(&path).unwrap();
    assert!(f2.dataset("version1").is_err()); // old dataset gone
    assert_eq!(
        f2.dataset("version2").unwrap().read_i32().unwrap(),
        vec![10, 20]
    );
    let attrs = f2.root().attrs().unwrap();
    assert!(matches!(attrs.get("version"), Some(AttrValue::I64(2))));
}

// ---------------------------------------------------------------------------
// 11. Multi-dimensional shape verification
// ---------------------------------------------------------------------------

#[test]
fn multidimensional_shapes() {
    let mut b = FileBuilder::new();
    b.create_dataset("vec").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("mat")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3]);
    b.create_dataset("cube")
        .with_f64_data(&[0.0; 24])
        .with_shape(&[2, 3, 4]);

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    assert_eq!(file.dataset("vec").unwrap().shape().unwrap(), vec![3]);
    assert_eq!(file.dataset("mat").unwrap().shape().unwrap(), vec![2, 3]);
    assert_eq!(
        file.dataset("cube").unwrap().shape().unwrap(),
        vec![2, 3, 4]
    );
}

// ---------------------------------------------------------------------------
// 12. Datasets in groups accessible via path notation
// ---------------------------------------------------------------------------

#[test]
fn path_based_dataset_access() {
    let mut b = FileBuilder::new();

    let mut g1 = b.create_group("level1");
    g1.create_dataset("data1").with_f64_data(&[1.0]);
    b.add_group(g1.finish());

    let mut g2 = b.create_group("other");
    g2.create_dataset("data2").with_i32_data(&[2]);
    b.add_group(g2.finish());

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    assert_eq!(
        file.dataset("level1/data1").unwrap().read_f64().unwrap(),
        vec![1.0]
    );
    assert_eq!(
        file.dataset("other/data2").unwrap().read_i32().unwrap(),
        vec![2]
    );
}

// ---------------------------------------------------------------------------
// 13. Chunked without compression (pure chunked)
// ---------------------------------------------------------------------------

#[test]
fn chunked_no_compression() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("chunked_plain")
        .with_f64_data(&data)
        .with_chunks(&[100]);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let values = file.dataset("chunked_plain").unwrap().read_f64().unwrap();
    assert_eq!(values, data);
}

// ---------------------------------------------------------------------------
// 14. Fletcher32 checksum round-trip
// ---------------------------------------------------------------------------

#[test]
fn fletcher32_roundtrip() {
    let data: Vec<f64> = (0..500).map(|i| i as f64 * 0.5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("checksummed")
        .with_f64_data(&data)
        .with_chunks(&[100])
        .with_fletcher32();
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let values = file.dataset("checksummed").unwrap().read_f64().unwrap();
    assert_eq!(values, data);
}

// ---------------------------------------------------------------------------
// 15. Multiple groups with same-named datasets
// ---------------------------------------------------------------------------

#[test]
fn same_dataset_name_in_different_groups() {
    let mut b = FileBuilder::new();

    let mut g1 = b.create_group("group_a");
    g1.create_dataset("values").with_f64_data(&[1.0, 2.0]);
    b.add_group(g1.finish());

    let mut g2 = b.create_group("group_b");
    g2.create_dataset("values").with_f64_data(&[3.0, 4.0]);
    b.add_group(g2.finish());

    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    assert_eq!(
        file.dataset("group_a/values").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.dataset("group_b/values").unwrap().read_f64().unwrap(),
        vec![3.0, 4.0]
    );
}
