use criterion::{criterion_group, criterion_main, Criterion};
use purehdf5_format::file_writer::FileWriter;
use purehdf5_format::signature::find_signature;
use purehdf5_format::superblock::Superblock;
use purehdf5_format::object_header::ObjectHeader;
use purehdf5_format::group_v2::resolve_path_any;
use purehdf5_format::message_type::MessageType;
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read::{read_as_f64, read_raw_data};
use purehdf5_format::filter_pipeline::FilterPipeline;
use purehdf5_format::chunked_read::read_chunked_data;

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

// ---- Dense attribute benchmarks ----

fn bench_write_dense_attrs(c: &mut Criterion) {
    use purehdf5_format::file_writer::AttrValue;
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
    use purehdf5_format::file_writer::AttrValue;
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
            purehdf5_format::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap()
        })
    });
}

// ---- Provenance benchmark ----

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

// ---- Checksum benchmark ----

fn bench_jenkins_lookup3(c: &mut Criterion) {
    let data: Vec<u8> = (0..1_000_000u32).flat_map(|v| v.to_le_bytes()).collect();
    c.bench_function("jenkins_lookup3_4MB", |b| {
        b.iter(|| purehdf5_format::checksum::jenkins_lookup3(&data))
    });
}

// ---- SHA-256 benchmark ----

fn bench_sha256(c: &mut Criterion) {
    let data: Vec<u8> = (0..1_000_000u32).flat_map(|v| v.to_le_bytes()).collect();
    c.bench_function("sha256_4MB", |b| {
        b.iter(|| purehdf5_format::provenance::sha256_hex(&data))
    });
}

// ---- Superblock parse benchmark ----

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

criterion_group!(
    benches,
    bench_write_contiguous,
    bench_write_chunked,
    bench_write_chunked_deflate,
    bench_read_contiguous,
    bench_read_chunked,
    bench_read_chunked_deflate,
    bench_roundtrip_contiguous,
    bench_roundtrip_chunked_deflate,
    bench_write_dense_attrs,
    bench_read_dense_attrs,
    bench_write_provenance,
    bench_jenkins_lookup3,
    bench_sha256,
    bench_parse_superblock,
);
criterion_main!(benches);
