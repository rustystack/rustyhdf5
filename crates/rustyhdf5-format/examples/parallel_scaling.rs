use std::time::Instant;
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

fn read_dataset(bytes: &[u8]) -> usize {
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
    let vals = read_as_f64(&raw, &dt).unwrap();
    vals.len()
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());
    let n = 10_000_000;
    eprintln!("Generating 10M f64 deflate file...");
    let bytes = make_deflate_file(n);
    eprintln!("File size: {:.1} MB, threads={}", bytes.len() as f64 / 1e6, threads);
    let _ = read_dataset(&bytes); // warmup
    let iters = 10;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let len = read_dataset(&bytes);
        assert_eq!(len, n);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("threads={:>4} median={:.2}ms min={:.2}ms max={:.2}ms", threads, times[iters/2], times[0], times[iters-1]);
}
