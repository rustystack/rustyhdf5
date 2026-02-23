use rustyhdf5_format::attribute::{extract_attributes, extract_attributes_full, find_attribute};
use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::data_read::{read_as_f64, read_as_f32, read_as_i32, read_as_i64, read_as_u64, read_as_strings, read_raw_data, read_raw_data_full};
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::file_writer::{AttrValue, FileWriter};
use rustyhdf5_format::filter_pipeline::{FilterPipeline, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE};
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;

// ============================================================
// Helpers
// ============================================================

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

/// Helper: read any dataset (contiguous or chunked) as f64.
fn read_dataset_f64_any(bytes: &[u8], path: &str) -> Vec<f64> {
    let sig = find_signature(bytes).unwrap();
    let sb = Superblock::parse(bytes, sig).unwrap();
    let addr = resolve_path_any(bytes, &sb, path).unwrap();
    let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();

    match &dl {
        DataLayout::Chunked { .. } => {
            let pipeline = hdr.messages.iter()
                .find(|m| m.msg_type == MessageType::FilterPipeline)
                .map(|m| FilterPipeline::parse(&m.data).unwrap());
            let raw = rustyhdf5_format::chunked_read::read_chunked_data(
                bytes, &dl, &ds, &dt, pipeline.as_ref(), sb.offset_size, sb.length_size,
            ).unwrap();
            read_as_f64(&raw, &dt).unwrap()
        }
        _ => {
            let raw = read_raw_data(bytes, &dl, &ds, &dt).unwrap();
            read_as_f64(&raw, &dt).unwrap()
        }
    }
}

/// Helper: read any dataset as i32.
fn read_dataset_i32_any(bytes: &[u8], path: &str) -> Vec<i32> {
    let sig = find_signature(bytes).unwrap();
    let sb = Superblock::parse(bytes, sig).unwrap();
    let addr = resolve_path_any(bytes, &sb, path).unwrap();
    let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(bytes, &dl, &ds, &dt).unwrap();
    read_as_i32(&raw, &dt).unwrap()
}

// ============================================================
// Superblock & Object Header Parsing
// ============================================================

#[test]
fn parse_minimal_v2_fixture() {
    let data = include_bytes!("fixtures/minimal_v2.h5");
    let offset = find_signature(data).expect("signature not found in minimal_v2.h5");
    assert_eq!(offset, 0);
    let sb = Superblock::parse(data, offset).expect("failed to parse superblock");
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

// ============================================================
// Simple Dataset Reading
// ============================================================

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

// ============================================================
// Variable-Length Strings
// ============================================================

#[test]
fn vl_strings_h5_names_dataset() {
    let file_data = include_bytes!("fixtures/vl_strings.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let names_addr = resolve_path_any(file_data, &sb, "names").unwrap();
    let hdr = ObjectHeader::parse(file_data, names_addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    assert_eq!(dataspace.num_elements(), 3);

    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    match &datatype {
        Datatype::VariableLength { is_string, .. } => assert!(is_string),
        other => panic!("expected VL string type, got {other:?}"),
    }

    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();

    let strings = rustyhdf5_format::vl_data::read_vl_strings(
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

// ============================================================
// Filter Pipeline Integration Tests
// ============================================================

#[test]
fn fixture_chunked_deflate_filter_pipeline() {
    let file_data = include_bytes!("fixtures/chunked_deflate.h5");
    let fp = parse_filter_pipeline_from_fixture(file_data, "data");
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
    assert_eq!(fp.filters[0].client_data, vec![8]);
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

// ============================================================
// Chunked Dataset Integration Tests
// ============================================================

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

// ============================================================
// V4 Layout Index Type Tests
// ============================================================

#[test]
fn v4_single_chunk_read() {
    let file_data = include_bytes!("fixtures/v4_single_chunk.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "small");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn v4_single_chunk_deflate_read() {
    let file_data = include_bytes!("fixtures/v4_single_chunk_deflate.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "small");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn v4_implicit_read() {
    let file_data = include_bytes!("fixtures/v4_implicit.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "data");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 100);
    for i in 0..100 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}

#[test]
fn v4_fixed_array_read() {
    let file_data = include_bytes!("fixtures/v4_fixed_array.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "data");
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 100);
    for i in 0..100 {
        assert_eq!(values[i], i as f64, "mismatch at index {i}");
    }
}

#[test]
fn v4_2d_fixed_array_read() {
    let file_data = include_bytes!("fixtures/v4_2d.h5");
    let (raw, datatype, _) = read_chunked_dataset(file_data, "matrix");
    let values = read_as_f32(&raw, &datatype).unwrap();
    assert_eq!(values.len(), 60);
    for i in 0..60 {
        assert!((values[i] - i as f32).abs() < 1e-6, "mismatch at index {i}: got {}", values[i]);
    }
}

// ============================================================
// V1 Group Navigation Tests
// ============================================================

#[test]
fn v1_two_groups_resolve_group1() {
    let file_data = include_bytes!("fixtures/two_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "group1");
    assert!(addr.is_ok(), "should resolve group1 in two_groups.h5");
}

#[test]
fn v1_two_groups_resolve_dataset_in_group() {
    let file_data = include_bytes!("fixtures/two_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "group1/values");
    assert!(addr.is_ok(), "should resolve group1/values");
}

#[test]
fn v1_two_groups_read_group1_values() {
    let file_data = include_bytes!("fixtures/two_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "group1/values").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_i32(&raw, &datatype).unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn v1_two_groups_read_group2_temps() {
    let file_data = include_bytes!("fixtures/two_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "group2/temps").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_f32(&raw, &datatype).unwrap();
    assert!((values[0] - 98.6).abs() < 0.01);
    assert!((values[1] - 37.0).abs() < 0.01);
}

#[test]
fn v1_nested_groups_deep_path() {
    let file_data = include_bytes!("fixtures/nested_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "a/b/c/deep");
    assert!(addr.is_ok(), "should resolve a/b/c/deep in nested_groups.h5");
}

#[test]
fn v1_nested_groups_read_deep_dataset() {
    let file_data = include_bytes!("fixtures/nested_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "a/b/c/deep").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![42.0]);
}

// ============================================================
// V2 Group Navigation Tests
// ============================================================

#[test]
fn v2_groups_resolve_sensor_temperature() {
    let file_data = include_bytes!("fixtures/v2_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "sensors/temperature");
    assert!(addr.is_ok(), "should resolve sensors/temperature in v2_groups.h5");
}

#[test]
fn v2_groups_read_temperature_values() {
    let file_data = include_bytes!("fixtures/v2_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "sensors/temperature").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![22.5, 23.1, 21.8]);
}

#[test]
fn v2_groups_read_humidity_values() {
    let file_data = include_bytes!("fixtures/v2_groups.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "sensors/humidity").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_i32(&raw, &datatype).unwrap();
    assert_eq!(values, vec![45, 50, 55]);
}

#[test]
fn v2_many_links_resolve_dataset() {
    let file_data = include_bytes!("fixtures/v2_many_links.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();
    let addr = resolve_path_any(file_data, &sb, "dataset_015").unwrap();
    let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let (datatype, _) = Datatype::parse(&dt_msg.data).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();

    let raw = read_raw_data(file_data, &layout, &dataspace, &datatype).unwrap();
    let values = read_as_f64(&raw, &datatype).unwrap();
    assert_eq!(values, vec![15.0]);
}

// ============================================================
// Write Round-Trip: All Datatype Classes
// ============================================================

#[test]
fn write_roundtrip_f64_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .with_shape(&[5]);
    let bytes = fw.finish().unwrap();
    let values = read_dataset_f64_any(&bytes, "data");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn write_roundtrip_f32_dataset() {
    let mut fw = FileWriter::new();
    let data: Vec<f32> = vec![1.5, 2.5, 3.5];
    fw.create_dataset("data")
        .with_f32_data(&data)
        .with_shape(&[3]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(&bytes, &dl, &ds, &dt).unwrap();
    let values = read_as_f32(&raw, &dt).unwrap();
    assert_eq!(values, vec![1.5, 2.5, 3.5]);
}

#[test]
fn write_roundtrip_i32_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("ints")
        .with_i32_data(&[10, 20, 30, 40]);
    let bytes = fw.finish().unwrap();
    let values = read_dataset_i32_any(&bytes, "ints");
    assert_eq!(values, vec![10, 20, 30, 40]);
}

#[test]
fn write_roundtrip_i64_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("longs")
        .with_i64_data(&[100, 200, 300]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "longs").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(&bytes, &dl, &ds, &dt).unwrap();
    let values = read_as_i64(&raw, &dt).unwrap();
    assert_eq!(values, vec![100, 200, 300]);
}

#[test]
fn write_roundtrip_u8_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("bytes")
        .with_u8_data(&[0, 127, 255]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "bytes").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(&bytes, &dl, &ds, &dt).unwrap();
    assert_eq!(raw, vec![0, 127, 255]);
}

// ============================================================
// Write Round-Trip: Attributes
// ============================================================

#[test]
fn write_roundtrip_scalar_f64_attr() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("scale", AttrValue::F64(3.14));
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let scale = find_attribute(&attrs, "scale").expect("scale attr not found");
    let vals = scale.read_as_f64().unwrap();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 3.14).abs() < 1e-10);
}

#[test]
fn write_roundtrip_string_attr() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("name", AttrValue::String("test_dataset".into()));
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let name = find_attribute(&attrs, "name").expect("name attr not found");
    let s = name.read_as_string().unwrap();
    assert_eq!(s, "test_dataset");
}

#[test]
fn write_roundtrip_i64_attr() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("version", AttrValue::I64(42));
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let ver = find_attribute(&attrs, "version").expect("version attr not found");
    let vals = ver.read_as_i64().unwrap();
    assert_eq!(vals, vec![42]);
}

#[test]
fn write_roundtrip_f64_array_attr() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0])
        .set_attr("temps", AttrValue::F64Array(vec![22.5, 23.1, 21.8]));
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let temps = find_attribute(&attrs, "temps").expect("temps attr not found");
    let vals = temps.read_as_f64().unwrap();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 22.5).abs() < 1e-10);
    assert!((vals[1] - 23.1).abs() < 1e-10);
    assert!((vals[2] - 21.8).abs() < 1e-10);
}

#[test]
fn write_roundtrip_root_attr() {
    let mut fw = FileWriter::new();
    fw.set_root_attr("file_format", AttrValue::String("HDF5".into()));
    fw.create_dataset("dummy").with_f64_data(&[0.0]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let hdr = ObjectHeader::parse(&bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let fmt = find_attribute(&attrs, "file_format").expect("file_format attr not found");
    let s = fmt.read_as_string().unwrap();
    assert_eq!(s, "HDF5");
}

// ============================================================
// Write Round-Trip: Chunked + Filters
// ============================================================

#[test]
fn write_roundtrip_chunked_no_filter() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[50])
        .with_chunks(&[10]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "data");
    assert_eq!(result, data);
}

#[test]
fn write_roundtrip_chunked_deflate() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[100])
        .with_chunks(&[25])
        .with_deflate(6);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "data");
    assert_eq!(result, data);
}

#[test]
fn write_roundtrip_chunked_shuffle_deflate() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[100])
        .with_chunks(&[50])
        .with_shuffle()
        .with_deflate(6);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "data");
    assert_eq!(result, data);
}

#[test]
fn write_roundtrip_chunked_fletcher32() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[100])
        .with_chunks(&[100])
        .with_fletcher32();
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "data");
    assert_eq!(result, data);
}

// ============================================================
// Write Round-Trip: Resizable Datasets (Extensible Array)
// ============================================================

#[test]
fn write_roundtrip_resizable_dataset() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[30])
        .with_chunks(&[10])
        .with_maxshape(&[u64::MAX]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "data");
    assert_eq!(result, data);
}

// ============================================================
// Write Round-Trip: Groups
// ============================================================

#[test]
fn write_roundtrip_group_with_dataset() {
    let mut fw = FileWriter::new();
    let mut grp = fw.create_group("mygroup");
    grp.create_dataset("vals")
        .with_f64_data(&[10.0, 20.0, 30.0])
        .with_shape(&[3]);
    let g = grp.finish();
    fw.add_group(g);
    let bytes = fw.finish().unwrap();
    let values = read_dataset_f64_any(&bytes, "mygroup/vals");
    assert_eq!(values, vec![10.0, 20.0, 30.0]);
}

#[test]
fn write_roundtrip_multiple_datasets() {
    let mut fw = FileWriter::new();
    fw.create_dataset("a").with_f64_data(&[1.0, 2.0]);
    fw.create_dataset("b").with_f64_data(&[3.0, 4.0]);
    fw.create_dataset("c").with_f64_data(&[5.0, 6.0]);
    let bytes = fw.finish().unwrap();
    assert_eq!(read_dataset_f64_any(&bytes, "a"), vec![1.0, 2.0]);
    assert_eq!(read_dataset_f64_any(&bytes, "b"), vec![3.0, 4.0]);
    assert_eq!(read_dataset_f64_any(&bytes, "c"), vec![5.0, 6.0]);
}

// ============================================================
// Write Round-Trip: 2D datasets
// ============================================================

#[test]
fn write_roundtrip_2d_contiguous() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    fw.create_dataset("matrix")
        .with_f64_data(&data)
        .with_shape(&[3, 4]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "matrix");
    assert_eq!(result, data);
}

#[test]
fn write_roundtrip_2d_chunked() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    fw.create_dataset("matrix")
        .with_f64_data(&data)
        .with_shape(&[4, 6])
        .with_chunks(&[2, 3]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "matrix");
    assert_eq!(result, data);
}

// ============================================================
// Edge Cases
// ============================================================

#[test]
fn write_roundtrip_single_element_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("scalar")
        .with_f64_data(&[42.0])
        .with_shape(&[1]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "scalar");
    assert_eq!(result, vec![42.0]);
}

#[test]
fn write_roundtrip_large_dataset() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    fw.create_dataset("big")
        .with_f64_data(&data)
        .with_shape(&[10_000]);
    let bytes = fw.finish().unwrap();
    let result = read_dataset_f64_any(&bytes, "big");
    assert_eq!(result.len(), 10_000);
    assert_eq!(result[0], 0.0);
    assert_eq!(result[9999], 9999.0);
}

#[test]
fn write_roundtrip_group_with_attrs() {
    let mut fw = FileWriter::new();
    let mut grp = fw.create_group("experiment");
    grp.set_attr("name", AttrValue::String("run_001".into()));
    grp.create_dataset("data")
        .with_f64_data(&[1.0, 2.0])
        .with_shape(&[2]);
    let g = grp.finish();
    fw.add_group(g);
    let bytes = fw.finish().unwrap();

    // Verify dataset
    let result = read_dataset_f64_any(&bytes, "experiment/data");
    assert_eq!(result, vec![1.0, 2.0]);

    // Verify group attribute
    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let grp_addr = resolve_path_any(&bytes, &sb, "experiment").unwrap();
    let hdr = ObjectHeader::parse(&bytes, grp_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();
    let name = find_attribute(&attrs, "name").expect("name attr not found");
    let s = name.read_as_string().unwrap();
    assert_eq!(s, "run_001");
}

#[test]
fn write_roundtrip_dataset_with_multiple_attrs() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .with_shape(&[3])
        .set_attr("description", AttrValue::String("test data".into()))
        .set_attr("version", AttrValue::I64(2))
        .set_attr("scale", AttrValue::F64(0.5));
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes(&hdr, sb.length_size).unwrap();

    let desc = find_attribute(&attrs, "description").expect("description not found");
    assert_eq!(desc.read_as_string().unwrap(), "test data");

    let ver = find_attribute(&attrs, "version").expect("version not found");
    assert_eq!(ver.read_as_i64().unwrap(), vec![2]);

    let scale = find_attribute(&attrs, "scale").expect("scale not found");
    let vals = scale.read_as_f64().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-10);
}

#[test]
fn superblock_v3_written_files() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data").with_f64_data(&[1.0]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    assert_eq!(sb.version, 3, "FileWriter should produce v3 superblocks");
    assert_eq!(sb.offset_size, 8);
    assert_eq!(sb.length_size, 8);
}

#[test]
fn write_roundtrip_large_chunked_i32() {
    let mut fw = FileWriter::new();
    let data: Vec<i32> = (0..5000).map(|i| i).collect();
    fw.create_dataset("big")
        .with_i32_data(&data)
        .with_shape(&[5000])
        .with_chunks(&[500]);
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "big").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let pipeline = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .map(|m| FilterPipeline::parse(&m.data).unwrap());
    let raw = read_raw_data_full(&bytes, &dl, &ds, &dt, pipeline.as_ref(), sb.offset_size, sb.length_size).unwrap();
    let values = read_as_i32(&raw, &dt).unwrap();
    assert_eq!(values.len(), 5000);
    assert_eq!(values[0], 0);
    assert_eq!(values[4999], 4999);
}

// ============================================================
// Dense Attribute Tests (AttributeInfo + fractal heap + B-tree v2)
// ============================================================

#[test]
fn dense_attrs_dataset_count() {
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();

    // Should have an AttributeInfo message since >8 attrs triggers dense storage
    let has_attr_info = hdr.messages.iter().any(|m| m.msg_type == MessageType::AttributeInfo);
    assert!(has_attr_info, "expected AttributeInfo message for dense attributes");

    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();
    assert_eq!(attrs.len(), 50, "expected 50 dense attributes");
}

#[test]
fn dense_attrs_dataset_first_attr() {
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();

    let attr_000 = find_attribute(&attrs, "attr_000").expect("attr_000 not found");
    let vals = attr_000.read_as_f64().unwrap();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 0.0).abs() < 1e-10, "attr_000 should be 0.0, got {}", vals[0]);
}

#[test]
fn dense_attrs_dataset_middle_attr() {
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();

    let attr_025 = find_attribute(&attrs, "attr_025").expect("attr_025 not found");
    let vals = attr_025.read_as_f64().unwrap();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 37.5).abs() < 1e-10, "attr_025 should be 37.5, got {}", vals[0]);
}

#[test]
fn dense_attrs_dataset_last_attr() {
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();

    let attr_049 = find_attribute(&attrs, "attr_049").expect("attr_049 not found");
    let vals = attr_049.read_as_f64().unwrap();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 73.5).abs() < 1e-10, "attr_049 should be 73.5, got {}", vals[0]);
}

#[test]
fn dense_attrs_dataset_all_values_correct() {
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();

    for i in 0..50 {
        let name = format!("attr_{i:03}");
        let attr = find_attribute(&attrs, &name).unwrap_or_else(|| panic!("{name} not found"));
        let vals = attr.read_as_f64().unwrap();
        let expected = i as f64 * 1.5;
        assert!(
            (vals[0] - expected).abs() < 1e-10,
            "{name}: expected {expected}, got {}",
            vals[0]
        );
    }
}

#[test]
fn dense_attrs_root_group() {
    let file_data = include_bytes!("fixtures/dense_attrs_root.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let hdr = ObjectHeader::parse(file_data, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();
    let attrs = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();
    assert_eq!(attrs.len(), 20, "expected 20 root group dense attributes");

    let attr_00 = find_attribute(&attrs, "root_attr_00").expect("root_attr_00 not found");
    let vals = attr_00.read_as_f64().unwrap();
    assert!((vals[0] - 0.0).abs() < 1e-10);

    let attr_19 = find_attribute(&attrs, "root_attr_19").expect("root_attr_19 not found");
    let vals = attr_19.read_as_f64().unwrap();
    assert!((vals[0] - 38.0).abs() < 1e-10);
}

#[test]
fn dense_attrs_compact_still_works() {
    // Ensure extract_attributes_full also works for compact (non-dense) attributes
    let file_data = include_bytes!("fixtures/attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();

    // extract_attributes_full should return the same results as extract_attributes for compact
    let attrs_compact = extract_attributes(&hdr, sb.length_size).unwrap();
    let attrs_full = extract_attributes_full(file_data, &hdr, sb.offset_size, sb.length_size).unwrap();
    assert_eq!(attrs_compact.len(), attrs_full.len());

    let desc = find_attribute(&attrs_full, "description").expect("description not found");
    assert_eq!(desc.read_as_string().unwrap(), "test dataset");
}

#[test]
fn dense_attrs_dataset_data_still_readable() {
    // Ensure the dataset data is still readable alongside dense attrs
    let file_data = include_bytes!("fixtures/dense_attrs.h5");
    let offset = find_signature(file_data).unwrap();
    let sb = Superblock::parse(file_data, offset).unwrap();

    let data_addr = resolve_path_any(file_data, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(file_data, data_addr as usize, sb.offset_size, sb.length_size).unwrap();

    let dt_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap();
    let ds_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap();
    let dl_msg = hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap();
    let (dt, _) = Datatype::parse(&dt_msg.data).unwrap();
    let ds = Dataspace::parse(&ds_msg.data, sb.length_size).unwrap();
    let dl = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size).unwrap();
    let raw = read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = read_as_f64(&raw, &dt).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

