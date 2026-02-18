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
