use criterion::{criterion_group, criterion_main, Criterion};
use rustyhdf5_format::file_writer::{AttrValue, CompoundTypeBuilder, FileWriter};
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::data_read::{read_as_f64, read_raw_data};
use rustyhdf5_format::filter_pipeline::FilterPipeline;
use rustyhdf5_format::chunked_read::read_chunked_data;

const N: usize = 1_000_000;

fn make_data() -> Vec<f64> {
    (0..N).map(|i| i as f64).collect()
}

fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
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
            let raw = read_chunked_data(
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

// ===========================================================================
// Write benchmarks
// ===========================================================================

fn bench_write_contiguous(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("write_1M_f64_contiguous", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64]);
            fw.finish().unwrap()
        })
    });
}

fn bench_write_chunked(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("write_1M_f64_chunked", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64])
                .with_chunks(&[10_000]);
            fw.finish().unwrap()
        })
    });
}

fn bench_write_chunked_deflate(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("write_1M_f64_chunked_deflate", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64])
                .with_chunks(&[10_000])
                .with_deflate(6);
            fw.finish().unwrap()
        })
    });
}

// ===========================================================================
// Read benchmarks
// ===========================================================================

fn bench_read_contiguous(c: &mut Criterion) {
    let data = make_data();
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[N as u64]);
    let bytes = fw.finish().unwrap();

    c.bench_function("read_1M_f64_contiguous", |b| {
        b.iter(|| read_dataset_f64(&bytes, "data"))
    });
}

fn bench_read_chunked(c: &mut Criterion) {
    let data = make_data();
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[10_000]);
    let bytes = fw.finish().unwrap();

    c.bench_function("read_1M_f64_chunked", |b| {
        b.iter(|| read_dataset_f64(&bytes, "data"))
    });
}

fn bench_read_chunked_deflate(c: &mut Criterion) {
    let data = make_data();
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[10_000])
        .with_deflate(6);
    let bytes = fw.finish().unwrap();

    c.bench_function("read_1M_f64_chunked_deflate", |b| {
        b.iter(|| read_dataset_f64(&bytes, "data"))
    });
}

// ===========================================================================
// Roundtrip benchmarks
// ===========================================================================

fn bench_roundtrip_contiguous(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("roundtrip_1M_f64_contiguous", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64]);
            let bytes = fw.finish().unwrap();
            read_dataset_f64(&bytes, "data")
        })
    });
}

fn bench_roundtrip_chunked_deflate(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("roundtrip_1M_f64_chunked_deflate", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64])
                .with_chunks(&[10_000])
                .with_deflate(6);
            let bytes = fw.finish().unwrap();
            read_dataset_f64(&bytes, "data")
        })
    });
}

// ===========================================================================
// Dense attribute benchmarks
// ===========================================================================

fn bench_write_dense_attrs(c: &mut Criterion) {
    c.bench_function("write_dataset_20_attrs_dense", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            let ds = fw.create_dataset("data");
            ds.with_f64_data(&[1.0, 2.0, 3.0]);
            for i in 0..20 {
                ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64));
            }
            fw.finish().unwrap()
        })
    });
}

fn bench_read_dense_attrs(c: &mut Criterion) {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0, 2.0, 3.0]);
    for i in 0..20 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64));
    }
    let bytes = fw.finish().unwrap();

    c.bench_function("read_dataset_20_attrs_dense", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
            let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
            rustyhdf5_format::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap()
        })
    });
}

// ===========================================================================
// 50-attribute read benchmark
// ===========================================================================

fn bench_write_50_attrs(c: &mut Criterion) {
    c.bench_function("write_dataset_50_attrs", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            let ds = fw.create_dataset("data");
            ds.with_f64_data(&[1.0]);
            for i in 0..50 {
                ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64));
            }
            fw.finish().unwrap()
        })
    });
}

fn bench_read_50_attrs(c: &mut Criterion) {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0]);
    for i in 0..50 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64));
    }
    let bytes = fw.finish().unwrap();

    c.bench_function("read_dataset_50_attrs", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
            let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
            rustyhdf5_format::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap()
        })
    });
}

// ===========================================================================
// Object header parse benchmark (complex header with many messages)
// ===========================================================================

fn bench_parse_object_header(c: &mut Criterion) {
    // Create a dataset with many features to produce a complex object header:
    // datatype, dataspace, data layout, filter pipeline, fill value,
    // plus multiple attributes — generating 10+ header messages.
    let data = make_data();
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[10_000])
        .with_deflate(6)
        .with_shuffle()
        .with_fletcher32();
    for i in 0..10 {
        ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64));
    }
    let bytes = fw.finish().unwrap();

    let sig = find_signature(&bytes).unwrap();
    let sb = Superblock::parse(&bytes, sig).unwrap();
    let addr = resolve_path_any(&bytes, &sb, "data").unwrap();

    c.bench_function("parse_object_header_complex", |b| {
        b.iter(|| {
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap()
        })
    });
}

// ===========================================================================
// Group navigation benchmark (100 datasets, resolve path to last)
// ===========================================================================

fn bench_group_navigation_100(c: &mut Criterion) {
    let mut fw = FileWriter::new();
    let mut grp = fw.create_group("grp");
    for i in 0..100 {
        grp.create_dataset(&format!("ds_{i:04}"))
            .with_f64_data(&[i as f64]);
    }
    fw.add_group(grp.finish());
    let bytes = fw.finish().unwrap();

    c.bench_function("group_nav_100_datasets", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            resolve_path_any(&bytes, &sb, "grp/ds_0099").unwrap()
        })
    });
}

// ===========================================================================
// String attribute benchmarks
// ===========================================================================

fn bench_write_string_attrs(c: &mut Criterion) {
    c.bench_function("write_10K_string_attrs", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            let ds = fw.create_dataset("data");
            ds.with_f64_data(&[1.0]);
            for i in 0..100 {
                ds.set_attr(
                    &format!("s{i:04}"),
                    AttrValue::String(format!("string_value_{i:08}")),
                );
            }
            fw.finish().unwrap()
        })
    });
}

fn bench_read_string_attrs(c: &mut Criterion) {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0]);
    for i in 0..100 {
        ds.set_attr(
            &format!("s{i:04}"),
            AttrValue::String(format!("string_value_{i:08}")),
        );
    }
    let bytes = fw.finish().unwrap();

    c.bench_function("read_100_string_attrs", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
            let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
            rustyhdf5_format::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap()
        })
    });
}

// ===========================================================================
// Compound type benchmarks (10K rows)
// ===========================================================================

fn bench_write_compound_10k(c: &mut Criterion) {
    let ct = CompoundTypeBuilder::new()
        .f64_field("x")
        .f64_field("y")
        .f64_field("z")
        .i32_field("id")
        .build();

    // Each row: 3 x f64 (24 bytes) + 1 x i32 (4 bytes) = 28 bytes
    let row_size = 28usize;
    let num_rows = 10_000u64;
    let mut raw = Vec::with_capacity(row_size * num_rows as usize);
    for i in 0..num_rows as usize {
        raw.extend_from_slice(&(i as f64).to_le_bytes());
        raw.extend_from_slice(&(i as f64 * 2.0).to_le_bytes());
        raw.extend_from_slice(&(i as f64 * 3.0).to_le_bytes());
        raw.extend_from_slice(&(i as i32).to_le_bytes());
    }

    c.bench_function("write_compound_10K_rows", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("table")
                .with_compound_data(ct.clone(), raw.clone(), num_rows)
                .with_shape(&[num_rows]);
            fw.finish().unwrap()
        })
    });
}

fn bench_read_compound_10k(c: &mut Criterion) {
    let ct = CompoundTypeBuilder::new()
        .f64_field("x")
        .f64_field("y")
        .f64_field("z")
        .i32_field("id")
        .build();

    let row_size = 28usize;
    let num_rows = 10_000u64;
    let mut raw = Vec::with_capacity(row_size * num_rows as usize);
    for i in 0..num_rows as usize {
        raw.extend_from_slice(&(i as f64).to_le_bytes());
        raw.extend_from_slice(&(i as f64 * 2.0).to_le_bytes());
        raw.extend_from_slice(&(i as f64 * 3.0).to_le_bytes());
        raw.extend_from_slice(&(i as i32).to_le_bytes());
    }

    let mut fw = FileWriter::new();
    fw.create_dataset("table")
        .with_compound_data(ct, raw, num_rows)
        .with_shape(&[num_rows]);
    let bytes = fw.finish().unwrap();

    c.bench_function("read_compound_10K_rows", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            let addr = resolve_path_any(&bytes, &sb, "table").unwrap();
            let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
            let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
            let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
            let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
            let (dt, _) = Datatype::parse(dt_data).unwrap();
            let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
            let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
            read_raw_data(&bytes, &dl, &ds, &dt).unwrap()
        })
    });
}

// ===========================================================================
// Provenance benchmark
// ===========================================================================

fn bench_write_provenance(c: &mut Criterion) {
    let data = make_data();
    c.bench_function("write_1M_f64_provenance", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("data")
                .with_f64_data(&data)
                .with_shape(&[N as u64])
                .with_provenance("bench", "2026-02-19T00:00:00Z", None);
            fw.finish().unwrap()
        })
    });
}

// ===========================================================================
// Checksum & hash benchmarks
// ===========================================================================

fn bench_jenkins_lookup3(c: &mut Criterion) {
    let data: Vec<u8> = (0..1_000_000u32).flat_map(|v| v.to_le_bytes()).collect();
    c.bench_function("jenkins_lookup3_4MB", |b| {
        b.iter(|| rustyhdf5_format::checksum::jenkins_lookup3(&data))
    });
}

fn bench_sha256(c: &mut Criterion) {
    let data: Vec<u8> = (0..1_000_000u32).flat_map(|v| v.to_le_bytes()).collect();
    c.bench_function("sha256_4MB", |b| {
        b.iter(|| rustyhdf5_format::provenance::sha256_hex(&data))
    });
}

// ===========================================================================
// Superblock parse benchmark
// ===========================================================================

fn bench_parse_superblock(c: &mut Criterion) {
    let mut fw = FileWriter::new();
    fw.create_dataset("data").with_f64_data(&[1.0]);
    let bytes = fw.finish().unwrap();
    c.bench_function("parse_superblock", |b| {
        b.iter(|| {
            let sig = find_signature(&bytes).unwrap();
            Superblock::parse(&bytes, sig).unwrap()
        })
    });
}

// ===========================================================================
// Multi-type write benchmark (i32, f32, i64 datasets)
// ===========================================================================

fn bench_write_multi_type(c: &mut Criterion) {
    let f32_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let i32_data: Vec<i32> = (0..N).map(|i| i as i32).collect();
    c.bench_function("write_1M_mixed_types", |b| {
        b.iter(|| {
            let mut fw = FileWriter::new();
            fw.create_dataset("f32").with_f32_data(&f32_data).with_shape(&[N as u64]);
            fw.create_dataset("i32").with_i32_data(&i32_data).with_shape(&[N as u64]);
            fw.finish().unwrap()
        })
    });
}

criterion_group!(
    benches,
    // Write
    bench_write_contiguous,
    bench_write_chunked,
    bench_write_chunked_deflate,
    // Read
    bench_read_contiguous,
    bench_read_chunked,
    bench_read_chunked_deflate,
    // Roundtrip
    bench_roundtrip_contiguous,
    bench_roundtrip_chunked_deflate,
    // Attributes (dense)
    bench_write_dense_attrs,
    bench_read_dense_attrs,
    bench_write_50_attrs,
    bench_read_50_attrs,
    // Object header
    bench_parse_object_header,
    // Group navigation
    bench_group_navigation_100,
    // String attributes
    bench_write_string_attrs,
    bench_read_string_attrs,
    // Compound type
    bench_write_compound_10k,
    bench_read_compound_10k,
    // Provenance & hashing
    bench_write_provenance,
    bench_jenkins_lookup3,
    bench_sha256,
    // Parsing
    bench_parse_superblock,
    // Multi-type
    bench_write_multi_type,
);
criterion_main!(benches);
