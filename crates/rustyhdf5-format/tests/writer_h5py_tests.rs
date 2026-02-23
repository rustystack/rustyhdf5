//! h5py round-trip tests for the file writer.
//!
//! These tests write HDF5 files with our writer and verify h5py can read them
//! (and vice versa). They require python3 + h5py to be installed.

use rustyhdf5_format::file_writer::{AttrValue, CompoundTypeBuilder, EnumTypeBuilder, FileWriter};

fn h5py_read(_path: &std::path::Path, script: &str) -> String {
    let o = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3");
    if !o.status.success() {
        panic!("h5py: {}", String::from_utf8_lossy(&o.stderr));
    }
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

// ---- h5py round-trip: basic datasets ----

#[test]
fn h5py_reads_our_f64_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]).with_shape(&[3]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_f64.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['data'][:].tolist()))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn h5py_reads_our_i32_dataset() {
    let mut fw = FileWriter::new();
    fw.create_dataset("ints").with_i32_data(&[10, 20, 30]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_i32.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['ints'][:].tolist()))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let values: Vec<i32> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn h5py_reads_dataset_with_attrs() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0, 2.0])
        .set_attr("scale", AttrValue::F64(0.5));
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_attrs.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'data': d[:].tolist(), 'scale': float(d.attrs['scale'])}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["data"], serde_json::json!([1.0, 2.0]));
    assert_eq!(v["scale"], serde_json::json!(0.5));
}

#[test]
fn h5py_reads_group_with_dataset() {
    let mut fw = FileWriter::new();
    let mut gb = fw.create_group("grp");
    gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
    fw.add_group(gb.finish());
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_grp.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['grp/vals'][:].tolist()))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(values, vec![10.0, 20.0]);
}

#[test]
fn h5py_reads_root_attrs() {
    let mut fw = FileWriter::new();
    fw.set_root_attr("version", AttrValue::I64(42));
    fw.create_dataset("dummy").with_f64_data(&[0.0]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_root_attrs.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); print(int(f.attrs['version']))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    assert_eq!(stdout, "42");
}

#[test]
fn h5py_reads_multiple_datasets() {
    let mut fw = FileWriter::new();
    fw.create_dataset("a").with_f64_data(&[1.0]);
    fw.create_dataset("b").with_f64_data(&[2.0]);
    fw.create_dataset("c").with_f64_data(&[3.0]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_multi.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); print(json.dumps({{k: f[k][:].tolist() for k in ['a','b','c']}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["a"], serde_json::json!([1.0]));
    assert_eq!(v["b"], serde_json::json!([2.0]));
    assert_eq!(v["c"], serde_json::json!([3.0]));
}

// ---- Compound / Enum / Array h5py round-trips ----

#[test]
fn h5py_reads_our_compound_dataset() {
    let ct = CompoundTypeBuilder::new()
        .f64_field("x")
        .f64_field("y")
        .i32_field("id")
        .build();
    let mut raw = Vec::new();
    raw.extend_from_slice(&1.5f64.to_le_bytes());
    raw.extend_from_slice(&2.5f64.to_le_bytes());
    raw.extend_from_slice(&10i32.to_le_bytes());
    raw.extend_from_slice(&3.5f64.to_le_bytes());
    raw.extend_from_slice(&4.5f64.to_le_bytes());
    raw.extend_from_slice(&20i32.to_le_bytes());

    let mut fw = FileWriter::new();
    fw.create_dataset("particles").with_compound_data(ct, raw, 2);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_compound.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['particles']; print(json.dumps({{'x':d['x'].tolist(),'y':d['y'].tolist(),'id':d['id'].tolist()}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["x"], serde_json::json!([1.5, 3.5]));
    assert_eq!(v["y"], serde_json::json!([2.5, 4.5]));
    assert_eq!(v["id"], serde_json::json!([10, 20]));
}

#[test]
fn h5py_reads_our_enum_dataset() {
    let et = EnumTypeBuilder::i32_based()
        .value("RED", 0)
        .value("GREEN", 1)
        .value("BLUE", 2)
        .build();

    let mut fw = FileWriter::new();
    fw.create_dataset("colors").with_enum_i32_data(et, &[1, 0, 2]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_enum.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['colors']; print(json.dumps(d[:].tolist()))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let values: Vec<i32> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(values, vec![1, 0, 2]);
}

#[test]
fn h5py_reads_our_array_dataset() {
    let mut raw = Vec::new();
    for v in &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
        raw.extend_from_slice(&v.to_le_bytes());
    }

    let mut fw = FileWriter::new();
    fw.create_dataset("vectors").with_array_data(
        rustyhdf5_format::type_builders::make_f64_type(), &[3], raw, 2,
    );
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_array.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['vectors']; print(json.dumps({{'shape':list(d.shape),'dtype':str(d.dtype),'values':d[:].tolist()}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["shape"], serde_json::json!([2]));
    assert_eq!(v["values"], serde_json::json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
}

#[test]
fn read_h5py_generated_compound() {
    let path = std::env::temp_dir().join("rustyhdf5_h5py_compound.h5");
    let gen_script = format!(
        r#"
import h5py, numpy as np
dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('id', 'i4')])
data = np.array([(1.0, 2.0, 10), (3.0, 4.0, 20)], dtype=dt)
f = h5py.File('{}', 'w', libver='latest')
f.create_dataset('particles', data=data)
f.close()
"#,
        path.display()
    );
    h5py_read(&path, &gen_script);

    let bytes = std::fs::read(&path).unwrap();
    let sig = rustyhdf5_format::signature::find_signature(&bytes).unwrap();
    let sb = rustyhdf5_format::superblock::Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "particles").unwrap();
    let hdr = rustyhdf5_format::object_header::ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::DataLayout).unwrap().data;
    let (dt, _) = rustyhdf5_format::datatype::Datatype::parse(dt_data).unwrap();
    let ds = rustyhdf5_format::dataspace::Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = rustyhdf5_format::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = rustyhdf5_format::data_read::read_raw_data(&bytes, &dl, &ds, &dt).unwrap();
    let fields = rustyhdf5_format::data_read::read_compound_fields(&raw, &dt).unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "x");
    let x_vals = rustyhdf5_format::data_read::read_as_f64(&fields[0].raw_data, &fields[0].datatype).unwrap();
    assert_eq!(x_vals, vec![1.0, 3.0]);
}

#[test]
fn read_h5py_generated_enum() {
    let path = std::env::temp_dir().join("rustyhdf5_h5py_enum.h5");
    let gen_script = format!(
        r#"
import h5py, numpy as np
dt = h5py.enum_dtype({{"RED": 0, "GREEN": 1, "BLUE": 2}}, basetype=np.int32)
data = np.array([1, 0, 2, 1], dtype=np.int32)
f = h5py.File('{}', 'w', libver='latest')
f.create_dataset('colors', data=data, dtype=dt)
f.close()
"#,
        path.display()
    );
    h5py_read(&path, &gen_script);

    let bytes = std::fs::read(&path).unwrap();
    let sig = rustyhdf5_format::signature::find_signature(&bytes).unwrap();
    let sb = rustyhdf5_format::superblock::Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "colors").unwrap();
    let hdr = rustyhdf5_format::object_header::ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::DataLayout).unwrap().data;
    let (dt, _) = rustyhdf5_format::datatype::Datatype::parse(dt_data).unwrap();
    let ds = rustyhdf5_format::dataspace::Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = rustyhdf5_format::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = rustyhdf5_format::data_read::read_raw_data(&bytes, &dl, &ds, &dt).unwrap();
    let names = rustyhdf5_format::data_read::read_enum_names(&raw, &dt).unwrap();
    assert_eq!(names, vec!["GREEN", "RED", "BLUE", "GREEN"]);
}

// ---- h5py chunked round-trip tests ----

#[test]
fn h5py_reads_chunked_no_compression() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[100]).with_chunks(&[20]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_chunked_nocomp.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks)}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["chunks"], serde_json::json!([20]));
}

#[test]
fn h5py_reads_chunked_deflate() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[100]).with_chunks(&[20]).with_deflate(6);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_chunked_deflate.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks),'compression':d.compression}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["compression"], serde_json::json!("gzip"));
}

#[test]
fn h5py_reads_chunked_shuffle_deflate() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[100]).with_chunks(&[50]).with_shuffle().with_deflate(6);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_chunked_shuffle_deflate.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'shuffle':bool(d.shuffle),'compression':d.compression}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["shuffle"], serde_json::json!(true));
    assert_eq!(v["compression"], serde_json::json!("gzip"));
}

#[test]
fn h5py_reads_chunked_fletcher32() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[100]).with_chunks(&[100]).with_fletcher32();
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_chunked_fletcher32.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'fletcher32':bool(d.fletcher32)}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["fletcher32"], serde_json::json!(true));
}

#[test]
fn h5py_reads_chunked_2d() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[4, 6]).with_chunks(&[2, 3]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_chunked_2d.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'shape':list(d.shape),'chunks':list(d.chunks),'values':d[:].flatten().tolist()}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["shape"], serde_json::json!([4, 6]));
    assert_eq!(v["chunks"], serde_json::json!([2, 3]));
}

#[test]
fn h5py_reads_2d_data() {
    let mut fw = FileWriter::new();
    fw.create_dataset("matrix").with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).with_shape(&[2, 3]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_test_2d.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py, json; f=h5py.File('{}','r'); d=f['matrix']; print(json.dumps({{'shape': list(d.shape), 'data': d[:].flatten().tolist()}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["shape"], serde_json::json!([2, 3]));
    assert_eq!(v["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

// ---- Extensible Array / resizable dataset h5py tests ----

#[test]
fn h5py_reads_resizable_dataset() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[50]).with_chunks(&[10]).with_maxshape(&[u64::MAX]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_ea_resizable.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks),'maxshape':list(None if x is None else x for x in d.maxshape)}}))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    assert_eq!(values, data);
    assert_eq!(v["chunks"], serde_json::json!([10]));
}

#[test]
fn read_h5py_generated_ea_file() {
    let path = std::env::temp_dir().join("rustyhdf5_h5py_ea.h5");
    let gen_script = format!(
        "import h5py,numpy as np; f=h5py.File('{}','w'); d=f.create_dataset('data',data=np.arange(30,dtype='float64'),chunks=(10,),maxshape=(None,)); f.close()",
        path.display()
    );
    h5py_read(&path, &gen_script);

    let bytes = std::fs::read(&path).unwrap();
    let sig = rustyhdf5_format::signature::find_signature(&bytes).unwrap();
    let sb = rustyhdf5_format::superblock::Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "data").unwrap();
    let hdr = rustyhdf5_format::object_header::ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    let dt_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Datatype).unwrap().data;
    let ds_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::Dataspace).unwrap().data;
    let dl_data = &hdr.messages.iter().find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::DataLayout).unwrap().data;
    let (dt, _) = rustyhdf5_format::datatype::Datatype::parse(dt_data).unwrap();
    let ds = rustyhdf5_format::dataspace::Dataspace::parse(ds_data, sb.length_size).unwrap();
    let dl = rustyhdf5_format::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
    let raw = match &dl {
        rustyhdf5_format::data_layout::DataLayout::Chunked { .. } => {
            let pipeline = hdr.messages.iter()
                .find(|m| m.msg_type == rustyhdf5_format::message_type::MessageType::FilterPipeline)
                .map(|m| rustyhdf5_format::filter_pipeline::FilterPipeline::parse(&m.data).unwrap());
            rustyhdf5_format::chunked_read::read_chunked_data(&bytes, &dl, &ds, &dt, pipeline.as_ref(), sb.offset_size, sb.length_size).unwrap()
        }
        _ => rustyhdf5_format::data_read::read_raw_data(&bytes, &dl, &ds, &dt).unwrap(),
    };
    let result = rustyhdf5_format::data_read::read_as_f64(&raw, &dt).unwrap();
    let expected: Vec<f64> = (0..30).map(|i| i as f64).collect();
    assert_eq!(result, expected);
}

#[test]
fn h5py_append_and_verify() {
    let mut fw = FileWriter::new();
    let initial: Vec<f64> = (0..10).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&initial).with_shape(&[10]).with_chunks(&[10]).with_maxshape(&[u64::MAX]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_ea_append.h5");
    std::fs::write(&path, &bytes).unwrap();

    let script = format!(
        r#"
import h5py, json, numpy as np
f = h5py.File('{}', 'a')
d = f['data']
for batch in range(3):
    old_size = d.shape[0]
    new_data = np.arange(old_size, old_size + 10, dtype='float64')
    d.resize(old_size + 10, axis=0)
    d[old_size:] = new_data
f.close()
f = h5py.File('{}', 'r')
d = f['data']
result = d[:].tolist()
shape = list(d.shape)
f.close()
print(json.dumps({{'values': result, 'shape': shape}}))
"#,
        path.display(),
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
    let expected: Vec<f64> = (0..40).map(|i| i as f64).collect();
    assert_eq!(values, expected);
    assert_eq!(v["shape"], serde_json::json!([40]));
}

#[test]
fn h5py_reads_resizable_single_chunk() {
    let mut fw = FileWriter::new();
    let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
    fw.create_dataset("data").with_f64_data(&data).with_shape(&[5]).with_chunks(&[10]).with_maxshape(&[u64::MAX]);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_ea_single.h5");
    std::fs::write(&path, &bytes).unwrap();
    let script = format!(
        "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps(d[:].tolist()))",
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
    assert_eq!(values, data);
}

// ---- Dense attribute h5py round-trip ----

#[test]
fn h5py_reads_dense_attrs() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0, 2.0, 3.0]);
    for i in 0..20 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
    }
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_dense_attrs.h5");
    std::fs::write(&path, &bytes).unwrap();

    let script = format!(
        r#"
import h5py, json
f = h5py.File('{}', 'r')
d = f['data']
attrs = {{k: float(v) for k, v in d.attrs.items()}}
data = d[:].tolist()
f.close()
print(json.dumps({{'data': data, 'num_attrs': len(attrs), 'attr_000': attrs.get('attr_000'), 'attr_010': attrs.get('attr_010'), 'attr_019': attrs.get('attr_019')}}))
"#,
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["data"], serde_json::json!([1.0, 2.0, 3.0]));
    assert_eq!(v["num_attrs"], serde_json::json!(20));
    assert_eq!(v["attr_000"], serde_json::json!(0.0));
    assert_eq!(v["attr_010"], serde_json::json!(15.0));
    assert_eq!(v["attr_019"], serde_json::json!(28.5));
}

#[test]
fn h5py_reads_50_dense_attrs() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[42.0]);
    for i in 0..50 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
    }
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_dense_50_attrs.h5");
    std::fs::write(&path, &bytes).unwrap();

    let script = format!(
        r#"
import h5py, json
f = h5py.File('{}', 'r')
d = f['data']
attrs = {{k: float(v) for k, v in d.attrs.items()}}
f.close()
print(json.dumps({{'num_attrs': len(attrs), 'first': attrs.get('attr_000'), 'last': attrs.get('attr_049')}}))
"#,
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["num_attrs"], serde_json::json!(50));
    assert_eq!(v["first"], serde_json::json!(0.0));
    assert_eq!(v["last"], serde_json::json!(73.5));
}

// ---- SHINES provenance h5py round-trips ----

#[test]
fn h5py_reads_provenance_attrs() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("sensor");
    ds.with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_provenance("rustyhdf5/test", "2026-02-19T12:00:00Z", Some("bench_42"));
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_provenance.h5");
    std::fs::write(&path, &bytes).unwrap();

    let script = format!(
        r#"
import h5py, json, hashlib, struct
f = h5py.File('{}', 'r')
d = f['sensor']
data = d[:].tolist()
sha = d.attrs['_provenance_sha256']
if isinstance(sha, bytes):
    sha = sha.decode('utf-8')
sha = sha.rstrip('\x00')
creator = d.attrs['_provenance_creator']
if isinstance(creator, bytes):
    creator = creator.decode('utf-8')
creator = creator.rstrip('\x00')
ts = d.attrs['_provenance_timestamp']
if isinstance(ts, bytes):
    ts = ts.decode('utf-8')
ts = ts.rstrip('\x00')
source = d.attrs['_provenance_source']
if isinstance(source, bytes):
    source = source.decode('utf-8')
source = source.rstrip('\x00')
# Verify SHA-256 matches the raw little-endian f64 bytes
raw = struct.pack('<4d', *data)
expected = hashlib.sha256(raw).hexdigest()
f.close()
print(json.dumps({{'data': data, 'sha256': sha, 'expected_sha256': expected, 'creator': creator, 'timestamp': ts, 'source': source}}))
"#,
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0]));
    assert_eq!(v["sha256"], v["expected_sha256"]);
    assert_eq!(v["creator"], serde_json::json!("rustyhdf5/test"));
    assert_eq!(v["timestamp"], serde_json::json!("2026-02-19T12:00:00Z"));
    assert_eq!(v["source"], serde_json::json!("bench_42"));
}

#[test]
fn h5py_reads_provenance_no_source() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_i32_data(&[10, 20, 30])
        .with_provenance("test-writer", "2026-01-01T00:00:00Z", None);
    let bytes = fw.finish().unwrap();
    let path = std::env::temp_dir().join("rustyhdf5_provenance_nosrc.h5");
    std::fs::write(&path, &bytes).unwrap();

    let script = format!(
        r#"
import h5py, json, hashlib, struct
f = h5py.File('{}', 'r')
d = f['data']
attr_names = sorted(d.attrs.keys())
sha = d.attrs['_provenance_sha256']
if isinstance(sha, bytes):
    sha = sha.decode('utf-8')
sha = sha.rstrip('\x00')
raw = struct.pack('<3i', *d[:].tolist())
expected = hashlib.sha256(raw).hexdigest()
has_source = '_provenance_source' in d.attrs
f.close()
print(json.dumps({{'attrs': attr_names, 'sha_ok': sha == expected, 'has_source': has_source}}))
"#,
        path.display()
    );
    let stdout = h5py_read(&path, &script);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["sha_ok"], serde_json::json!(true));
    assert_eq!(v["has_source"], serde_json::json!(false));
}

#[test]
fn provenance_verify_written_file() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("values");
    ds.with_f64_data(&[100.0, 200.0, 300.0])
        .with_provenance("integrity-test", "2026-02-19T00:00:00Z", None);
    let bytes = fw.finish().unwrap();

    // Use our verification API to check integrity
    let sig = rustyhdf5_format::signature::find_signature(&bytes).unwrap();
    let sb = rustyhdf5_format::superblock::Superblock::parse(&bytes, sig).unwrap();
    let addr = rustyhdf5_format::group_v2::resolve_path_any(&bytes, &sb, "values").unwrap();
    let hdr = rustyhdf5_format::object_header::ObjectHeader::parse(
        &bytes, addr as usize, sb.offset_size, sb.length_size,
    ).unwrap();
    let result = rustyhdf5_format::provenance::verify_dataset(
        &bytes, &hdr, sb.offset_size, sb.length_size,
    ).unwrap();
    assert_eq!(result, rustyhdf5_format::provenance::VerifyResult::Ok);
}
