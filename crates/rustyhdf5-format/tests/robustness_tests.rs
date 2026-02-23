//! Robustness tests: verify parsers return errors (not panics) on malformed input.

use rustyhdf5_format::error::FormatError;
use rustyhdf5_format::superblock::Superblock;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::fractal_heap::FractalHeapHeader;
use rustyhdf5_format::btree_v2::BTreeV2Header;
use rustyhdf5_format::signature;

// ---- Truncated / empty inputs ----

#[test]
fn empty_file_no_signature() {
    let result = signature::find_signature(&[]);
    assert!(result.is_err());
}

#[test]
fn short_file_no_signature() {
    let result = signature::find_signature(&[0x89, 0x48, 0x44]);
    assert!(result.is_err());
}

#[test]
fn truncated_superblock() {
    // Valid signature but truncated superblock
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
    data.push(3); // version 3
    // Truncated — not enough bytes for a full superblock
    let sig = signature::find_signature(&data).unwrap();
    let result = Superblock::parse(&data, sig);
    assert!(result.is_err());
}

#[test]
fn truncated_object_header_v2() {
    // "OHDR" + version 2 but truncated
    let data = b"OHDR\x02\x00";
    let result = ObjectHeader::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

#[test]
fn truncated_datatype() {
    let result = Datatype::parse(&[0x03]); // class 0, version 1, but truncated
    assert!(result.is_err());
}

#[test]
fn truncated_dataspace() {
    let result = Dataspace::parse(&[0x02], 8); // version but truncated
    assert!(result.is_err());
}

#[test]
fn truncated_fractal_heap() {
    let data = b"FRHP\x00";
    let result = FractalHeapHeader::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

#[test]
fn truncated_btree_v2() {
    let data = b"BTHD\x00";
    let result = BTreeV2Header::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

// ---- Invalid signatures ----

#[test]
fn bad_object_header_signature() {
    let data = b"XHDR\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = ObjectHeader::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

#[test]
fn bad_fractal_heap_signature() {
    let data = b"XRHP\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = FractalHeapHeader::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

#[test]
fn bad_btree_v2_signature() {
    let data = b"XTHD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = BTreeV2Header::parse(data, 0, 8, 8);
    assert!(result.is_err());
}

// ---- Invalid versions ----

#[test]
fn bad_superblock_version() {
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
    data.push(99); // unsupported version
    data.extend_from_slice(&[0; 100]); // padding
    let sig = signature::find_signature(&data).unwrap();
    let result = Superblock::parse(&data, sig);
    assert!(result.is_err());
}

#[test]
fn bad_fractal_heap_version() {
    let mut data = vec![0; 256];
    data[0..4].copy_from_slice(b"FRHP");
    data[4] = 99; // bad version
    let result = FractalHeapHeader::parse(&data, 0, 8, 8);
    assert!(matches!(result, Err(FormatError::InvalidFractalHeapVersion(99))));
}

#[test]
fn bad_btree_v2_version() {
    let mut data = vec![0; 64];
    data[0..4].copy_from_slice(b"BTHD");
    data[4] = 99; // bad version
    let result = BTreeV2Header::parse(&data, 0, 8, 8);
    assert!(matches!(result, Err(FormatError::InvalidBTreeV2Version(99))));
}

// ---- Random / garbage data ----

#[test]
fn random_bytes_no_signature() {
    let garbage: Vec<u8> = (0..1024).map(|i| (i * 37 + 13) as u8).collect();
    let result = signature::find_signature(&garbage);
    assert!(result.is_err());
}

#[test]
fn random_bytes_no_panic_superblock() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    // Should not panic even if it fails
    let _ = Superblock::parse(&garbage, 0);
}

#[test]
fn random_bytes_no_panic_object_header() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    let _ = ObjectHeader::parse(&garbage, 0, 8, 8);
}

#[test]
fn random_bytes_no_panic_datatype() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    let _ = Datatype::parse(&garbage);
}

#[test]
fn random_bytes_no_panic_dataspace() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    let _ = Dataspace::parse(&garbage, 8);
}

#[test]
fn random_bytes_no_panic_fractal_heap() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    let _ = FractalHeapHeader::parse(&garbage, 0, 8, 8);
}

#[test]
fn random_bytes_no_panic_btree_v2() {
    let garbage: Vec<u8> = (0..256).map(|i| (i * 73 + 5) as u8).collect();
    let _ = BTreeV2Header::parse(&garbage, 0, 8, 8);
}

// ---- Multiple garbage patterns (pseudo-fuzz) ----

#[test]
fn pseudo_fuzz_superblock() {
    for seed in 0u16..100 {
        let data: Vec<u8> = (0..128).map(|i| ((i as u16 * seed.wrapping_add(7)) & 0xFF) as u8).collect();
        let _ = signature::find_signature(&data);
        let _ = Superblock::parse(&data, 0);
    }
}

#[test]
fn pseudo_fuzz_object_header() {
    for seed in 0u16..100 {
        let data: Vec<u8> = (0..512).map(|i| ((i as u16 * seed.wrapping_add(13)) & 0xFF) as u8).collect();
        let _ = ObjectHeader::parse(&data, 0, 8, 8);
        let _ = ObjectHeader::parse(&data, 0, 4, 4);
    }
}

#[test]
fn pseudo_fuzz_datatype() {
    for seed in 0u16..100 {
        let data: Vec<u8> = (0..128).map(|i| ((i as u16 * seed.wrapping_add(3)) & 0xFF) as u8).collect();
        let _ = Datatype::parse(&data);
    }
}

#[test]
fn pseudo_fuzz_fractal_heap() {
    for seed in 0u16..100 {
        let data: Vec<u8> = (0..512).map(|i| ((i as u16 * seed.wrapping_add(17)) & 0xFF) as u8).collect();
        let _ = FractalHeapHeader::parse(&data, 0, 8, 8);
    }
}

#[test]
fn pseudo_fuzz_btree_v2() {
    for seed in 0u16..100 {
        let data: Vec<u8> = (0..256).map(|i| ((i as u16 * seed.wrapping_add(11)) & 0xFF) as u8).collect();
        let _ = BTreeV2Header::parse(&data, 0, 8, 8);
    }
}

// ---- Checksum corruption detection ----

#[test]
fn corrupted_file_detected() {
    // Write a valid file then corrupt it
    let mut fw = rustyhdf5_format::file_writer::FileWriter::new();
    fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
    let mut bytes = fw.finish().unwrap();

    // Verify it works uncorrupted
    let sig = signature::find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let _ = ObjectHeader::parse(
        &bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size,
    ).unwrap();

    // Corrupt a byte in the middle of the object header
    let oh_start = sb.root_group_address as usize;
    if oh_start + 20 < bytes.len() {
        bytes[oh_start + 10] ^= 0xFF;
        // Should detect corruption via checksum
        let result = ObjectHeader::parse(
            &bytes, oh_start, sb.offset_size, sb.length_size,
        );
        assert!(result.is_err(), "corrupted object header should fail checksum");
    }
}

// ---- Attribute reading robustness ----

#[test]
fn extract_attrs_from_dataset_no_panic() {
    // Build a valid file, then try extracting attrs from the dataset
    let mut fw = rustyhdf5_format::file_writer::FileWriter::new();
    fw.create_dataset("data").with_f64_data(&[1.0]);
    let bytes = fw.finish().unwrap();
    let sig = signature::find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    // Should return empty vec, not panic
    let attrs = rustyhdf5_format::attribute::extract_attributes_full(
        &bytes, &hdr, sb.offset_size, sb.length_size,
    ).unwrap();
    assert!(attrs.is_empty());
}

// ---- Provenance verification on corrupted data ----

#[test]
fn provenance_mismatch_on_corruption() {
    let mut fw = rustyhdf5_format::file_writer::FileWriter::new();
    let ds = fw.create_dataset("sensor");
    ds.with_f64_data(&[1.0, 2.0, 3.0])
        .with_provenance("test", "2026-01-01T00:00:00Z", None);
    let mut bytes = fw.finish().unwrap();

    // Corrupt the raw data region (last bytes in file)
    let len = bytes.len();
    bytes[len - 5] ^= 0xFF;

    let sig = signature::find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "sensor").unwrap();
    let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let result = rustyhdf5_format::provenance::verify_dataset(
        &bytes, &hdr, sb.offset_size, sb.length_size,
    ).unwrap();
    assert!(
        matches!(result, rustyhdf5_format::provenance::VerifyResult::Mismatch { .. }),
        "corrupted data should produce hash mismatch"
    );
}
