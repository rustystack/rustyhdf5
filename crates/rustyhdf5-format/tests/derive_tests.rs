//! Tests for the `#[derive(H5Type)]` proc macro.

use rustyhdf5_derive::H5Type;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::data_read::{read_as_f64, read_as_i32, read_compound_fields};

// ---- Test structs ----

#[derive(H5Type, Debug, PartialEq)]
struct SimpleF64 {
    x: f64,
    y: f64,
}

#[derive(H5Type, Debug, PartialEq)]
struct MixedNumerics {
    a: f64,
    b: f32,
    c: i32,
    d: u16,
}

#[derive(H5Type, Debug, PartialEq)]
struct AllIntegers {
    a: i8,
    b: i16,
    c: i32,
    d: i64,
    e: u8,
    f: u16,
    g: u32,
    h: u64,
}

#[derive(H5Type, Debug, PartialEq)]
struct WithBool {
    flag: bool,
    value: f64,
}

#[derive(H5Type, Debug, PartialEq)]
struct WithArray {
    coords: [f64; 3],
    id: i32,
}

#[derive(H5Type, Debug, PartialEq)]
struct SingleField {
    value: f64,
}

#[derive(H5Type, Debug, PartialEq)]
struct IntArrayStruct {
    data: [i32; 4],
}

#[derive(H5Type, Debug, PartialEq)]
struct U8ArrayStruct {
    bytes: [u8; 8],
}

#[derive(H5Type, Debug, PartialEq)]
struct ManyFields {
    a: f64,
    b: f32,
    c: i8,
    d: i16,
    e: i32,
    f: i64,
    g: u8,
    h: u16,
    i: u32,
    j: u64,
    k: bool,
}

// ---- Test 1: Simple struct datatype generation ----

#[test]
fn simple_f64_datatype() {
    let dt = SimpleF64::hdf5_datatype();
    match &dt {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 16); // 2 x f64
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "x");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "y");
            assert_eq!(members[1].byte_offset, 8);
        }
        _ => panic!("expected Compound datatype"),
    }
}

// ---- Test 2: Nested/mixed numerics datatype ----

#[test]
fn mixed_numerics_datatype() {
    let dt = MixedNumerics::hdf5_datatype();
    match &dt {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 18); // 8 + 4 + 4 + 2
            assert_eq!(members.len(), 4);
            assert_eq!(members[0].name, "a");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "b");
            assert_eq!(members[1].byte_offset, 8);
            assert_eq!(members[2].name, "c");
            assert_eq!(members[2].byte_offset, 12);
            assert_eq!(members[3].name, "d");
            assert_eq!(members[3].byte_offset, 16);
        }
        _ => panic!("expected Compound datatype"),
    }
}

// ---- Test 3: Round-trip serialization of simple struct ----

#[test]
fn simple_f64_roundtrip() {
    let s = SimpleF64 { x: 3.125, y: 2.75 };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 16);
    let s2 = SimpleF64::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 4: Round-trip of mixed numerics ----

#[test]
fn mixed_numerics_roundtrip() {
    let s = MixedNumerics {
        a: 1.0,
        b: 2.5,
        c: -42,
        d: 1000,
    };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 18);
    let s2 = MixedNumerics::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 5: All integer types ----

#[test]
fn all_integers_roundtrip() {
    let s = AllIntegers {
        a: -1,
        b: -256,
        c: -100_000,
        d: -1_000_000_000,
        e: 255,
        f: 65535,
        g: 4_000_000_000,
        h: 10_000_000_000,
    };
    let bytes = s.to_bytes();
    let expected_size = 1 + 2 + 4 + 8 + 1 + 2 + 4 + 8;
    assert_eq!(bytes.len(), expected_size);
    let s2 = AllIntegers::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 6: Bool field ----

#[test]
fn with_bool_roundtrip() {
    let s = WithBool {
        flag: true,
        value: 99.9,
    };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 9); // 1 + 8
    assert_eq!(bytes[0], 1); // true = 1
    let s2 = WithBool::from_bytes(&bytes);
    assert_eq!(s, s2);

    let s_false = WithBool {
        flag: false,
        value: 0.0,
    };
    let bytes2 = s_false.to_bytes();
    assert_eq!(bytes2[0], 0);
    let s3 = WithBool::from_bytes(&bytes2);
    assert_eq!(s_false, s3);
}

// ---- Test 7: Fixed-size array field ----

#[test]
fn with_array_roundtrip() {
    let s = WithArray {
        coords: [1.0, 2.0, 3.0],
        id: 42,
    };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 28); // 3*8 + 4
    let s2 = WithArray::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 8: Array datatype generation ----

#[test]
fn array_datatype_structure() {
    let dt = WithArray::hdf5_datatype();
    match &dt {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 28);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "coords");
            match &members[0].datatype {
                Datatype::Array {
                    base_type,
                    dimensions,
                } => {
                    assert_eq!(dimensions, &vec![3u32]);
                    assert!(matches!(**base_type, Datatype::FloatingPoint { size: 8, .. }));
                }
                _ => panic!("expected Array datatype for coords"),
            }
        }
        _ => panic!("expected Compound datatype"),
    }
}

// ---- Test 8b: Single field struct ----

#[test]
fn single_field_roundtrip() {
    let s = SingleField { value: 42.0 };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 8);
    let s2 = SingleField::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 9: Compound bytes compatible with data_read ----

#[test]
fn compound_read_compatibility() {
    let items = vec![
        SimpleF64 { x: 1.0, y: 10.0 },
        SimpleF64 { x: 2.0, y: 20.0 },
        SimpleF64 { x: 3.0, y: 30.0 },
    ];
    let mut raw = Vec::new();
    for item in &items {
        raw.extend_from_slice(&item.to_bytes());
    }

    let dt = SimpleF64::hdf5_datatype();
    let fields = read_compound_fields(&raw, &dt).unwrap();
    assert_eq!(fields.len(), 2);

    let x_vals = read_as_f64(&fields[0].raw_data, &fields[0].datatype).unwrap();
    assert_eq!(x_vals, vec![1.0, 2.0, 3.0]);

    let y_vals = read_as_f64(&fields[1].raw_data, &fields[1].datatype).unwrap();
    assert_eq!(y_vals, vec![10.0, 20.0, 30.0]);
}

// ---- Test 10: Round-trip through FileWriter ----

#[test]
fn roundtrip_through_file_writer() {
    use rustyhdf5_format::file_writer::FileWriter;
    use rustyhdf5_format::superblock::Superblock;
    use rustyhdf5_format::object_header::ObjectHeader;
    use rustyhdf5_format::signature::find_signature;
    use rustyhdf5_format::data_read::read_raw_data;

    let items = vec![
        MixedNumerics { a: 1.5, b: 2.5, c: -10, d: 100 },
        MixedNumerics { a: 3.0, b: 4.0, c: 20, d: 200 },
    ];

    let dt = MixedNumerics::hdf5_datatype();
    let mut raw_data = Vec::new();
    for item in &items {
        raw_data.extend_from_slice(&item.to_bytes());
    }

    let mut fw = FileWriter::new();
    fw.create_dataset("compound_ds")
        .with_compound_data(dt.clone(), raw_data.clone(), 2);
    let file_bytes = fw.finish().unwrap();

    // Parse the file back
    let sig_offset = find_signature(&file_bytes).unwrap();
    let sb = Superblock::parse(&file_bytes, sig_offset).unwrap();
    let oh = ObjectHeader::parse(&file_bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();

    // Find the compound dataset link
    let mut ds_addr = None;
    for msg in &oh.messages {
        if msg.msg_type == rustyhdf5_format::message_type::MessageType::Link {
            let link = rustyhdf5_format::link_message::LinkMessage::parse(&msg.data, sb.offset_size).unwrap();
            if link.name == "compound_ds" {
                if let rustyhdf5_format::link_message::LinkTarget::Hard { object_header_address } = link.link_target {
                    ds_addr = Some(object_header_address);
                }
            }
        }
    }
    let ds_addr = ds_addr.expect("compound_ds link not found");

    let ds_oh = ObjectHeader::parse(&file_bytes, ds_addr as usize, sb.offset_size, sb.length_size).unwrap();

    // Extract datatype, dataspace, and layout
    let mut found_dt = None;
    let mut found_ds = None;
    let mut found_layout = None;
    for msg in &ds_oh.messages {
        match msg.msg_type {
            rustyhdf5_format::message_type::MessageType::Datatype => {
                let (parsed_dt, _) = rustyhdf5_format::datatype::Datatype::parse(&msg.data).unwrap();
                found_dt = Some(parsed_dt);
            }
            rustyhdf5_format::message_type::MessageType::Dataspace => {
                found_ds = Some(rustyhdf5_format::dataspace::Dataspace::parse(&msg.data, sb.length_size).unwrap());
            }
            rustyhdf5_format::message_type::MessageType::DataLayout => {
                found_layout = Some(rustyhdf5_format::data_layout::DataLayout::parse(&msg.data, sb.offset_size, sb.length_size).unwrap());
            }
            _ => {}
        }
    }

    let parsed_dt = found_dt.expect("no datatype");
    let parsed_ds = found_ds.expect("no dataspace");
    let layout = found_layout.expect("no layout");

    let raw_back = read_raw_data(&file_bytes, &layout, &parsed_ds, &parsed_dt).unwrap();
    let fields = read_compound_fields(&raw_back, &parsed_dt).unwrap();

    let a_vals = read_as_f64(&fields[0].raw_data, &fields[0].datatype).unwrap();
    assert_eq!(a_vals, vec![1.5, 3.0]);

    let c_vals = read_as_i32(&fields[2].raw_data, &fields[2].datatype).unwrap();
    assert_eq!(c_vals, vec![-10, 20]);
}

// ---- Test 11: Many fields struct ----

#[test]
fn many_fields_roundtrip() {
    let s = ManyFields {
        a: 1.0,
        b: 2.0,
        c: -3,
        d: 400,
        e: -500,
        f: 6_000_000,
        g: 7,
        h: 800,
        i: 9_000,
        j: 10_000_000_000,
        k: true,
    };
    let bytes = s.to_bytes();
    let s2 = ManyFields::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 12: Integer array struct ----

#[test]
fn int_array_roundtrip() {
    let s = IntArrayStruct {
        data: [10, 20, -30, 40],
    };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 16); // 4 x i32
    let s2 = IntArrayStruct::from_bytes(&bytes);
    assert_eq!(s, s2);
}

// ---- Test 13: U8 array struct ----

#[test]
fn u8_array_roundtrip() {
    let s = U8ArrayStruct {
        bytes: [1, 2, 3, 4, 5, 6, 7, 8],
    };
    let bytes = s.to_bytes();
    assert_eq!(bytes.len(), 8);
    let s2 = U8ArrayStruct::from_bytes(&bytes);
    assert_eq!(s, s2);
}
