use purehdf5_format::attribute::{extract_attributes, find_attribute};
use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read::{read_as_f64, read_as_f32, read_as_i32, read_raw_data, read_raw_data_full};
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
use purehdf5_format::filter_pipeline::{FilterPipeline, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE};
use purehdf5_format::group_v2::resolve_path_any;
use purehdf5_format::message_type::MessageType;
use purehdf5_format::object_header::ObjectHeader;
use purehdf5_format::signature::find_signature;
use purehdf5_format::superblock::Superblock;

#[test]
fn parse_minimal_v2_fixture() {
    let data = include_bytes!("fixtures/minimal_v2.h5");
    let offset = find_signature(data).expect("signature not found in minimal_v2.h5");
    assert_eq!(offset, 0);
    let sb = Superblock::parse(data, offset).expect("failed to parse superblock");
    // h5py may create v0 or v2 depending on version; just verify it parses
    assert!(sb.version <= 3);
    assert_eq!(sb.offset_size, 8);
    assert_eq!(sb.length_size, 8);
    assert_eq!(sb.base_address, 0);
    assert!(sb.eof_address > 0);
    assert!(sb.root_group_address > 0);
}

#[test]
fn parse_minimal_v2_object_header() {
    let data = include_bytes!("fixtures/minimal_v2.h5");
    let offset = find_signature(data).expect("signature not found");
    let sb = Superblock::parse(data, offset).expect("failed to parse superblock");
    let root_addr = sb.root_group_address as usize;
    let hdr = ObjectHeader::parse(data, root_addr, sb.offset_size, sb.length_size)
        .expect("failed to parse root group object header");
    assert!(hdr.version == 1 || hdr.version == 2);
    // Root group should have at least one message (e.g., SymbolTable or Link)
    assert!(!hdr.messages.is_empty(), "root group object header has no messages");
}

#[test]
fn parse_simple_dataset_object_header() {
    let data = include_bytes!("fixtures/simple_dataset.h5");
    let offset = find_signature(data).expect("signature not found");
    let sb = Superblock::parse(data, offset).expect("failed to parse superblock");
    let root_addr = sb.root_group_address as usize;
    let hdr = ObjectHeader::parse(data, root_addr, sb.offset_size, sb.length_size)
        .expect("failed to parse root group object header");
    assert!(hdr.version == 1 || hdr.version == 2);
    assert!(!hdr.messages.is_empty(), "root group object header has no messages");
}

#[test]
fn parse_simple_dataset_fixture() {
    let data = include_bytes!("fixtures/simple_dataset.h5");
    let offset = find_signature(data).expect("signature not found in simple_dataset.h5");
    assert_eq!(offset, 0);
    let sb = Superblock::parse(data, offset).expect("failed to parse superblock");
    assert!(sb.version <= 3);
    assert_eq!(sb.offset_size, 8);
    assert!(sb.eof_address > 0);
    assert!(sb.root_group_address > 0);
}

#[test]
fn read_simple_dataset_values() {
    let file_data = include_bytes!("fixtures/simple_dataset.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let dataset_offset = 0x0320usize;
    let hdr = ObjectHeader::parse(file_data, dataset_offset, sb.offset_size, sb.length_size)
        .expect("failed to parse dataset object header");

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    assert_eq!(dataspace.dimensions, vec![3]);

    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    assert_eq!(datatype.type_size(), 8);

    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();
    match &layout {
        DataLayout::Contiguous { address, size } => {
            assert_eq!(*address, Some(0x0800));
            assert_eq!(*size, 24);
        }
        other => panic!("expected Contiguous, got {other:?}"),
    }

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

// ============================================================
// Attribute integration tests
// ============================================================

#[test]
fn attrs_h5_dataset_description() {
    let file_data = include_bytes!("fixtures/attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let desc = find_attribute(&attrs, "description").expect("description attr not found");
    let s = desc.read_as_string().unwrap();
    assert_eq!(s, "test dataset");
}

#[test]
fn attrs_h5_dataset_version() {
    let file_data = include_bytes!("fixtures/attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let version_attr = find_attribute(&attrs, "version").expect("version attr not found");
    let vals = version_attr.read_as_i64().unwrap();
    assert_eq!(vals, vec![42]);
}

#[test]
fn attrs_h5_dataset_scale() {
    let file_data = include_bytes!("fixtures/attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let scale_attr = find_attribute(&attrs, "scale").expect("scale attr not found");
    let vals = scale_attr.read_as_f64().unwrap();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 3.14).abs() < 1e-10);
}

#[test]
fn attrs_h5_root_file_type() {
    let file_data = include_bytes!("fixtures/attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let root_addr = sb.root_group_address as usize;
    let hdr = ObjectHeader::parse(file_data, root_addr, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let ft = find_attribute(&attrs, "file_type").expect("file_type attr not found");
    let s = ft.read_as_string().unwrap();
    assert_eq!(s, "experiment");
}

#[test]
fn mixed_attrs_h5_temperatures() {
    let file_data = include_bytes!("fixtures/mixed_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let exp_addr = resolve_path_any(file_data, &sb, "experiment").unwrap();
    let hdr = ObjectHeader::parse(file_data, exp_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let temp = find_attribute(&attrs, "temperatures").expect("temperatures attr not found");
    let vals = temp.read_as_f64().unwrap();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 22.5).abs() < 1e-10);
    assert!((vals[1] - 23.1).abs() < 1e-10);
    assert!((vals[2] - 21.8).abs() < 1e-10);
}

#[test]
fn mixed_attrs_h5_name() {
    let file_data = include_bytes!("fixtures/mixed_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let exp_addr = resolve_path_any(file_data, &sb, "experiment").unwrap();
    let hdr = ObjectHeader::parse(file_data, exp_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let name_attr = find_attribute(&attrs, "name").expect("name attr not found");
    let s = name_attr.read_as_string().unwrap();
    assert_eq!(s, "run_001");
}

#[test]
fn mixed_attrs_h5_iterations() {
    let file_data = include_bytes!("fixtures/mixed_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let exp_addr = resolve_path_any(file_data, &sb, "experiment").unwrap();
    let hdr = ObjectHeader::parse(file_data, exp_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let iter_attr = find_attribute(&attrs, "iterations").expect("iterations attr not found");
    let vals = iter_attr.read_as_i64().unwrap();
    assert_eq!(vals, vec![1000]);
}

#[test]
fn vl_strings_h5_names_dataset() {
    let file_data = include_bytes!("fixtures/vl_strings.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let names_addr = resolve_path_any(file_data, &sb, "names").unwrap();
    let hdr = ObjectHeader::parse(file_data, names_addr as usize, sb.offset_size, sb.length_size).unwrap();

    // Get dataspace
    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    assert_eq!(dataspace.num_elements(), 3);

    // Get datatype - should be VL string
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    match &datatype {
        Datatype::VariableLength { is_string, .. } => assert!(is_string),
        other => panic!("expected VL string type, got {other:?}"),
    }

    // Get layout and read raw data
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();

    // Resolve VL strings
    let strings = purehdf5_format::vl_data::read_vl_strings(
        file_data, &raw, dataspace.num_elements(), sb.offset_size, sb.length_size,
    ).unwrap();
    assert_eq!(strings, vec!["Alice", "Bob", "Charlie"]);
}

#[test]
fn vl_strings_h5_root_vl_attr() {
    let file_data = include_bytes!("fixtures/vl_strings.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let root_addr = sb.root_group_address as usize;
    let hdr = ObjectHeader::parse(file_data, root_addr, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let vl_attr = find_attribute(&attrs, "vl_attr").expect("vl_attr not found");
    let strings = vl_attr.read_vl_strings(file_data, sb.offset_size, sb.length_size).unwrap();
    assert_eq!(strings.len(), 1);
    assert_eq!(strings[0], "hello variable length");
}

// --- Filter Pipeline Integration Tests ---

/// Helper: navigate to dataset and find the FilterPipeline message.
fn parse_filter_pipeline_from_fixture(file_data: &[u8], dataset_path: &str) -> FilterPipeline {
    let offset = find_signature(file_data).expect("signature not found");
    let sb = Superblock::parse(file_data, offset).expect("failed to parse superblock");
    let addr = resolve_path_any(file_data, &sb, dataset_path).expect("dataset not found");
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
        .expect("failed to parse object header");
    let fp_msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .expect("no FilterPipeline message found");
    FilterPipeline::parse(&fp_msg.data).expect("failed to parse filter pipeline")
}

#[test]
fn fixture_chunked_deflate_filter_pipeline() {
    let file_data = include_bytes!("fixtures/chunked_deflate.h5");
    let fp = parse_filter_pipeline_from_fixture(file_data, "data");
    // h5py writes deflate with flags=1 (optional), name="deflate"
    assert!(fp.filters.iter().any(|f| f.filter_id == FILTER_DEFLATE));
    let deflate = fp.filters.iter().find(|f| f.filter_id == FILTER_DEFLATE).unwrap();
    assert_eq!(deflate.client_data, vec![6]);
}

#[test]
fn fixture_chunked_shuffle_deflate_filter_pipeline() {
    let file_data = include_bytes!("fixtures/chunked_shuffle_deflate.h5");
    let fp = parse_filter_pipeline_from_fixture(file_data, "data");
    assert_eq!(fp.filters.len(), 2);
    assert_eq!(fp.filters[0].filter_id, FILTER_SHUFFLE);
    assert_eq!(fp.filters[0].client_data, vec![8]); // element_size=8 for f64
    assert_eq!(fp.filters[1].filter_id, FILTER_DEFLATE);
    assert_eq!(fp.filters[1].client_data, vec![6]);
}

#[test]
fn fixture_chunked_fletcher32_filter_pipeline() {
    let file_data = include_bytes!("fixtures/chunked_fletcher32.h5");
    let fp = parse_filter_pipeline_from_fixture(file_data, "data");
    assert_eq!(fp.filters.len(), 1);
    assert_eq!(fp.filters[0].filter_id, FILTER_FLETCHER32);
}

#[test]
fn fixture_chunked_deflate_has_chunked_layout() {
    let file_data = include_bytes!("fixtures/chunked_deflate.h5");
    let offset = find_signature(file_data).expect("signature not found");
    let sb = Superblock::parse(file_data, offset).expect("failed to parse superblock");
    let addr = resolve_path_any(file_data, &sb, "data").expect("dataset not found");
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
        .expect("failed to parse object header");
    let layout_msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .expect("no DataLayout message found");
    let layout = DataLayout::parse(&layout_msg.data, sb.offset_size, sb.length_size)
        .expect("failed to parse data layout");
    match layout {
        DataLayout::Chunked { btree_address, .. } => {
            assert!(btree_address.is_some(), "btree_address should be set");
        }
        other => panic!("expected Chunked layout, got {:?}", other),
    }
}

// --- Chunked Dataset Integration Tests ---

/// Helper to read a chunked dataset from a fixture file.
fn read_chunked_dataset(file_data: &[u8], dataset_path: &str) -> (Vec<u8>, Datatype, Dataspace) {
    let offset = find_signature(file_data).expect("signature not found");
    let sb = Superblock::parse(file_data, offset).expect("failed to parse superblock");
    let addr = resolve_path_any(file_data, &sb, dataset_path).expect("dataset not found");
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
        .expect("failed to parse object header");

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();

    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();

    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let pipeline = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .map(|m| FilterPipeline::parse(&m.data).unwrap());

    let raw = read_raw_data_full(
        file_data, &layout, &dataspace, &datatype,
        pipeline.as_ref(), sb.offset_size, sb.length_size,
    ).unwrap();

    (raw, datatype, dataspace)
}

#[test]
fn chunked_deflate_read_values() {
    let file_data = include_bytes!("fixtures/chunked_deflate.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "data");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 100);
    for i in 0..100 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}

#[test]
fn chunked_shuffle_deflate_read_values() {
    let file_data = include_bytes!("fixtures/chunked_shuffle_deflate.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "data");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 100);
    for i in 0..100 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}

#[test]
fn chunked_fletcher32_read_values() {
    let file_data = include_bytes!("fixtures/chunked_fletcher32.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "data");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 100);
    for i in 0..100 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}

#[test]
fn chunked_2d_read_values() {
    let file_data = include_bytes!("fixtures/chunked_2d.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "matrix");
    let values = read_as_f32(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 60);
    for i in 0..60 {
        assert!((values[i] - i as f32).abs() < 1e-6, "mismatch at index {i}: got {}", values[i]);
    }
}

#[test]
fn chunked_large_read_values() {
    let file_data = include_bytes!("fixtures/chunked_large.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "big");
    let values = read_as_i32(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 1000);
    for i in 0..1000 {
        assert_eq!(values[i], i as i32, "mismatch at index {i}");
    }
}

#[test]
fn chunked_nofilter_read_values() {
    let file_data = include_bytes!("fixtures/chunked_nofilter.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "raw");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 50);
    for i in 0..50 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}
