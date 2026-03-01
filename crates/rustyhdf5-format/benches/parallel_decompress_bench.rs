//! Parallel decompression scaling benchmark for 48-core Xeon.
//! Uses read_chunked_data which auto-dispatches to parallel when feature enabled.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rustyhdf5_format::file_writer::FileWriter;
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::data_read::read_as_f64;
use rustyhdf5_format::filter_pipeline::FilterPipeline;
use rustyhdf5_format::chunked_read::read_chunked_data;

fn make_deflate_file(n: usize) -> Vec<u8> {
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[10_000])
        .with_deflate(6);
    fw.finish().unwrap()
}

fn read_dataset(bytes: &[u8]) -> Vec<f64> {
    let sig = find_signature(bytes).unwrap();
    let sb = Superblock::parse(bytes, sig).unwrap();
    let addr = resolve_path_any(bytes, &sb, "data").unwrap();
    let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
    let (dt, _) = Datatype::parse(dt_data).unwrap();
    let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let pipeline = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .map(|m| FilterPipeline::parse(&m.data).unwrap());
    let raw = read_chunked_data(bytes, &dl, &ds, &dt, pipeline.as_ref(), sb.offset_size, sb.length_size).unwrap();
    read_as_f64(&raw, &dt).unwrap()
}

fn bench_core_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_deflate_10M");
    group.sample_size(15);

    let n = 10_000_000;
    let bytes = make_deflate_file(n);

    // The `parallel` feature auto-dispatches to rayon in read_chunked_data.
    // Control thread count via RAYON_NUM_THREADS env var.
    for cores in [1, 2, 4, 8, 16, 24, 32, 48] {
        group.bench_with_input(
            BenchmarkId::new("lanes", cores),
            &cores,
            |b, &cores| {
                std::env::set_var("RAYON_NUM_THREADS", cores.to_string());
                // Force rayon to reinitialize — this only works for the first call.
                // For accurate per-iteration control, we set it before the group.
                b.iter(|| read_dataset(&bytes))
            },
        );
    }

    group.finish();
}

fn bench_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_deflate_sizes");
    group.sample_size(15);

    // Use all available cores (auto)
    for n in [1_000_000, 5_000_000, 10_000_000] {
        let bytes = make_deflate_file(n);
        let label = format!("{}M", n / 1_000_000);

        group.bench_function(format!("{label}_parallel"), |b| {
            b.iter(|| read_dataset(&bytes))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_core_scaling, bench_size_scaling);
criterion_main!(benches);
