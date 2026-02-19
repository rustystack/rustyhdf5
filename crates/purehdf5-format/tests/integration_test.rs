use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read::{read_as_f64, read_raw_data};
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
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
