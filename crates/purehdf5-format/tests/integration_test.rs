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
