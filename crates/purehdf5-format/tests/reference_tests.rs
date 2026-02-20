//! Integration tests for HDF5 reference type reading.

use purehdf5_format::data_read::{
    read_object_references, read_region_references,
};
use purehdf5_format::datatype::{Datatype, ReferenceType};

#[test]
fn object_ref_single_valid() {
    let dt = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let raw = 4096u64.to_le_bytes().to_vec();
    let refs = read_object_references(&raw, &dt, 8).unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].address, 4096);
    assert!(!refs[0].is_null());
}

#[test]
fn object_ref_null_detection() {
    let dt = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let raw = u64::MAX.to_le_bytes().to_vec();
    let refs = read_object_references(&raw, &dt, 8).unwrap();
    assert_eq!(refs.len(), 1);
    assert!(refs[0].is_null());
}

#[test]
fn object_ref_multiple() {
    let dt = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let mut raw = Vec::new();
    raw.extend_from_slice(&100u64.to_le_bytes());
    raw.extend_from_slice(&u64::MAX.to_le_bytes());
    raw.extend_from_slice(&200u64.to_le_bytes());
    raw.extend_from_slice(&300u64.to_le_bytes());
    raw.extend_from_slice(&u64::MAX.to_le_bytes());

    let refs = read_object_references(&raw, &dt, 8).unwrap();
    assert_eq!(refs.len(), 5);
    assert_eq!(refs[0].address, 100);
    assert!(refs[1].is_null());
    assert_eq!(refs[2].address, 200);
    assert_eq!(refs[3].address, 300);
    assert!(refs[4].is_null());
}

#[test]
fn object_ref_4byte_offset_size() {
    let dt = Datatype::Reference {
        size: 4,
        ref_type: ReferenceType::Object,
    };
    let mut raw = Vec::new();
    raw.extend_from_slice(&256u32.to_le_bytes());
    raw.extend_from_slice(&u32::MAX.to_le_bytes());

    let refs = read_object_references(&raw, &dt, 4).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].address, 256);
    assert!(refs[1].is_null());
}

#[test]
fn object_ref_wrong_type_errors() {
    let dt = Datatype::Reference {
        size: 12,
        ref_type: ReferenceType::DatasetRegion,
    };
    let raw = vec![0u8; 12];
    let err = read_object_references(&raw, &dt, 8).unwrap_err();
    assert!(matches!(
        err,
        purehdf5_format::error::FormatError::TypeMismatch { .. }
    ));
}

#[test]
fn object_ref_non_reference_type_errors() {
    let dt = Datatype::FixedPoint {
        size: 8,
        byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 64,
    };
    let raw = vec![0u8; 8];
    let err = read_object_references(&raw, &dt, 8).unwrap_err();
    assert!(matches!(
        err,
        purehdf5_format::error::FormatError::TypeMismatch { .. }
    ));
}

#[test]
fn object_ref_size_mismatch() {
    let dt = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let raw = vec![0u8; 7]; // not a multiple of 8
    let err = read_object_references(&raw, &dt, 8).unwrap_err();
    assert!(matches!(
        err,
        purehdf5_format::error::FormatError::DataSizeMismatch { .. }
    ));
}

#[test]
fn region_ref_basic() {
    let dt = Datatype::Reference {
        size: 12,
        ref_type: ReferenceType::DatasetRegion,
    };
    let raw: Vec<u8> = (0..24).collect();
    let refs = read_region_references(&raw, &dt).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].raw, (0u8..12).collect::<Vec<u8>>());
    assert_eq!(refs[1].raw, (12u8..24).collect::<Vec<u8>>());
}

#[test]
fn region_ref_wrong_type_errors() {
    let dt = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let raw = vec![0u8; 8];
    let err = read_region_references(&raw, &dt).unwrap_err();
    assert!(matches!(
        err,
        purehdf5_format::error::FormatError::TypeMismatch { .. }
    ));
}

#[test]
fn region_ref_size_mismatch() {
    let dt = Datatype::Reference {
        size: 12,
        ref_type: ReferenceType::DatasetRegion,
    };
    let raw = vec![0u8; 11]; // not a multiple of 12
    let err = read_region_references(&raw, &dt).unwrap_err();
    assert!(matches!(
        err,
        purehdf5_format::error::FormatError::DataSizeMismatch { .. }
    ));
}

// ---- h5py integration: read object references from an h5py-created file ----

#[test]
fn h5py_object_reference_roundtrip() {
    // Generate an HDF5 file with h5py containing object references
    let path = std::env::temp_dir().join("purehdf5_test_objrefs.h5");
    let script = format!(
        r#"
import h5py
import numpy as np

f = h5py.File('{}', 'w')
f.create_dataset('target_a', data=[1.0, 2.0, 3.0])
f.create_dataset('target_b', data=[10, 20, 30])
refs = [f['target_a'].ref, f['target_b'].ref]
f.create_dataset('refs', data=refs)
f.close()
print('ok')
"#,
        path.display()
    );
    let output = std::process::Command::new("python3")
        .args(["-c", &script])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => {
            eprintln!("skipping h5py_object_reference_roundtrip: python3+h5py not available");
            return;
        }
    };
    let stdout = String::from_utf8(output.stdout).unwrap();
    if !stdout.trim().contains("ok") {
        eprintln!("skipping h5py_object_reference_roundtrip: h5py script failed");
        return;
    }

    // Read the file and parse object references
    let file_data = std::fs::read(&path).unwrap();
    let sig_offset = purehdf5_format::signature::find_signature(&file_data).unwrap();
    let sb = purehdf5_format::superblock::Superblock::parse(&file_data, sig_offset).unwrap();
    let root_oh = purehdf5_format::object_header::ObjectHeader::parse(
        &file_data,
        sb.root_group_address as usize,
        sb.offset_size,
        sb.length_size,
    )
    .unwrap();

    // Find the 'refs' dataset
    let mut refs_addr = None;
    for msg in &root_oh.messages {
        if msg.msg_type == purehdf5_format::message_type::MessageType::Link {
            let link = purehdf5_format::link_message::LinkMessage::parse(
                &msg.data,
                sb.offset_size,
            )
            .unwrap();
            if link.name == "refs" {
                if let purehdf5_format::link_message::LinkTarget::Hard {
                    object_header_address,
                } = link.link_target
                {
                    refs_addr = Some(object_header_address);
                }
            }
        }
    }
    let refs_addr = refs_addr.expect("'refs' dataset link not found");

    let ds_oh = purehdf5_format::object_header::ObjectHeader::parse(
        &file_data,
        refs_addr as usize,
        sb.offset_size,
        sb.length_size,
    )
    .unwrap();

    let mut found_dt = None;
    let mut found_ds = None;
    let mut found_layout = None;
    for msg in &ds_oh.messages {
        match msg.msg_type {
            purehdf5_format::message_type::MessageType::Datatype => {
                let (dt, _) = purehdf5_format::datatype::Datatype::parse(&msg.data).unwrap();
                found_dt = Some(dt);
            }
            purehdf5_format::message_type::MessageType::Dataspace => {
                found_ds = Some(purehdf5_format::dataspace::Dataspace::parse(
                    &msg.data,
                    sb.length_size,
                )
                .unwrap());
            }
            purehdf5_format::message_type::MessageType::DataLayout => {
                found_layout = Some(purehdf5_format::data_layout::DataLayout::parse(
                    &msg.data,
                    sb.offset_size,
                    sb.length_size,
                )
                .unwrap());
            }
            _ => {}
        }
    }

    let ref_dt = found_dt.expect("no datatype in refs dataset");
    let ref_ds = found_ds.expect("no dataspace in refs dataset");
    let ref_layout = found_layout.expect("no layout in refs dataset");

    // Verify the datatype is an object reference
    match &ref_dt {
        Datatype::Reference { ref_type, .. } => {
            assert_eq!(*ref_type, ReferenceType::Object);
        }
        _ => panic!("expected Reference datatype, got {:?}", ref_dt),
    }

    let raw = purehdf5_format::data_read::read_raw_data(
        &file_data,
        &ref_layout,
        &ref_ds,
        &ref_dt,
    )
    .unwrap();

    let obj_refs = read_object_references(&raw, &ref_dt, sb.offset_size).unwrap();
    assert_eq!(obj_refs.len(), 2);

    // Both references should be non-null and point to valid addresses
    assert!(!obj_refs[0].is_null());
    assert!(!obj_refs[1].is_null());
    assert_ne!(obj_refs[0].address, obj_refs[1].address);

    // Clean up
    let _ = std::fs::remove_file(&path);
}
