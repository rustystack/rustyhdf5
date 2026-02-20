//! Bidirectional interop tests between purehdf5 and h5py (Python).
//!
//! Tests are skipped if python3 or h5py are not available.

use std::process::Command;

use purehdf5::{AttrValue, CompoundTypeBuilder, DType, File, FileBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn python_available() -> bool {
    Command::new("python3")
        .args(["-c", "import h5py; print(h5py.__version__)"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

macro_rules! skip_if_no_python {
    () => {
        if !python_available() {
            eprintln!("SKIP: python3 with h5py not available");
            return;
        }
    };
}

/// Run a Python script and panic if it fails.
fn run_python(script: &str) {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("failed to run python3");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Python script failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        );
    }
}

/// Run a Python script and return stdout as a trimmed string.
fn run_python_output(script: &str) -> String {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("failed to run python3");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python script failed:\nSTDERR: {stderr}");
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

// ===========================================================================
// Part A: purehdf5 writes -> h5py reads
// ===========================================================================

// ---------------------------------------------------------------------------
// A1. Write f64 dataset -> h5py reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn purehdf5_writes_f64_h5py_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64_test.h5");
    let path_str = path.display().to_string();

    let data = vec![1.5, 2.5, 3.5, -4.5, 0.0];
    let mut b = FileBuilder::new();
    b.create_dataset("values").with_f64_data(&data);
    b.write(&path).unwrap();

    let script = format!(
        r#"
import h5py, numpy as np, json
with h5py.File("{path_str}", "r") as f:
    ds = f["values"]
    assert ds.dtype == np.float64, f"expected float64, got {{ds.dtype}}"
    assert ds.shape == (5,), f"expected (5,), got {{ds.shape}}"
    vals = ds[()].tolist()
    print(json.dumps(vals))
"#
    );
    let output = run_python_output(&script);
    let vals: Vec<f64> = serde_json_minimal_parse(&output);
    assert_eq!(vals, data);
}

// ---------------------------------------------------------------------------
// A2. Write string dataset -> h5py reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn purehdf5_writes_strings_h5py_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strings_test.h5");
    let path_str = path.display().to_string();

    // purehdf5 doesn't have a direct with_string_data, so we write f64 and
    // test string attributes instead
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&[1.0]);
    b.set_attr("title", AttrValue::String("Hello World".into()));
    b.set_attr(
        "tags",
        AttrValue::StringArray(vec!["alpha".into(), "beta".into()]),
    );
    b.write(&path).unwrap();

    let script = format!(
        r#"
import h5py
with h5py.File("{path_str}", "r") as f:
    title = f.attrs["title"]
    if isinstance(title, bytes):
        title = title.decode()
    print(title)
    tags = f.attrs["tags"]
    tag_list = [t.decode() if isinstance(t, bytes) else t for t in tags]
    print(",".join(tag_list))
"#
    );
    let output = run_python_output(&script);
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines[0], "Hello World");
    assert_eq!(lines[1], "alpha,beta");
}

// ---------------------------------------------------------------------------
// A3. Write chunked+compressed dataset -> h5py reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn purehdf5_writes_chunked_compressed_h5py_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunked_test.h5");
    let path_str = path.display().to_string();

    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("compressed")
        .with_f64_data(&data)
        .with_chunks(&[100])
        .with_deflate(6);
    b.write(&path).unwrap();

    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File("{path_str}", "r") as f:
    ds = f["compressed"]
    assert ds.chunks is not None, "expected chunked dataset"
    assert ds.compression == "gzip", f"expected gzip, got {{ds.compression}}"
    vals = ds[()].tolist()
    assert len(vals) == 1000, f"expected 1000 values, got {{len(vals)}}"
    # Check first and last
    assert abs(vals[0] - 0.0) < 1e-10
    assert abs(vals[999] - 9.99) < 1e-10
    print("OK")
"#
    );
    let output = run_python_output(&script);
    assert_eq!(output, "OK");
}

// ---------------------------------------------------------------------------
// A4. Write groups with attributes -> h5py reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn purehdf5_writes_groups_attrs_h5py_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("groups_test.h5");
    let path_str = path.display().to_string();

    let mut b = FileBuilder::new();
    b.set_attr("file_version", AttrValue::I64(3));

    let mut g = b.create_group("experiment");
    g.set_attr("run_id", AttrValue::I64(42));
    g.set_attr("description", AttrValue::String("test run".into()));
    g.create_dataset("measurements")
        .with_f64_data(&[10.0, 20.0, 30.0]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    let script = format!(
        r#"
import h5py
with h5py.File("{path_str}", "r") as f:
    assert f.attrs["file_version"] == 3
    g = f["experiment"]
    assert g.attrs["run_id"] == 42
    desc = g.attrs["description"]
    if isinstance(desc, bytes):
        desc = desc.decode()
    assert desc == "test run", f"got {{desc}}"
    vals = g["measurements"][()].tolist()
    assert vals == [10.0, 20.0, 30.0]
    print("OK")
"#
    );
    let output = run_python_output(&script);
    assert_eq!(output, "OK");
}

// ---------------------------------------------------------------------------
// A5. Write compound type -> h5py reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn purehdf5_writes_compound_h5py_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compound_test.h5");
    let path_str = path.display().to_string();

    let dt = CompoundTypeBuilder::new()
        .f64_field("x")
        .f64_field("y")
        .i32_field("id")
        .build();

    let mut raw = Vec::new();
    for &(x, y, id) in &[(1.0f64, 2.0f64, 10i32), (3.0, 4.0, 20)] {
        raw.extend_from_slice(&x.to_le_bytes());
        raw.extend_from_slice(&y.to_le_bytes());
        raw.extend_from_slice(&id.to_le_bytes());
    }

    let mut b = FileBuilder::new();
    b.create_dataset("points").with_compound_data(dt, raw, 2);
    b.write(&path).unwrap();

    let script = format!(
        r#"
import h5py
with h5py.File("{path_str}", "r") as f:
    ds = f["points"]
    assert ds.dtype.names == ("x", "y", "id"), f"got {{ds.dtype.names}}"
    assert len(ds) == 2
    row0 = ds[0]
    assert abs(float(row0["x"]) - 1.0) < 1e-10
    assert abs(float(row0["y"]) - 2.0) < 1e-10
    assert int(row0["id"]) == 10
    row1 = ds[1]
    assert abs(float(row1["x"]) - 3.0) < 1e-10
    assert int(row1["id"]) == 20
    print("OK")
"#
    );
    let output = run_python_output(&script);
    assert_eq!(output, "OK");
}

// ===========================================================================
// Part B: h5py writes -> purehdf5 reads
// ===========================================================================

// ---------------------------------------------------------------------------
// B6. h5py creates f64 dataset -> purehdf5 reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn h5py_writes_f64_purehdf5_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5py_f64.h5");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File("{path_str}", "w") as f:
    f.create_dataset("values", data=np.array([1.5, 2.5, 3.5, -4.5, 0.0], dtype=np.float64))
"#
    );
    run_python(&script);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("values").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.shape().unwrap(), vec![5]);
    assert_eq!(ds.read_f64().unwrap(), vec![1.5, 2.5, 3.5, -4.5, 0.0]);
}

// ---------------------------------------------------------------------------
// B7. h5py creates string dataset -> purehdf5 reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn h5py_writes_strings_purehdf5_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5py_strings.h5");
    let path_str = path.display().to_string();

    // Use fixed-length strings (S10) since purehdf5 doesn't support vlen strings yet
    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File("{path_str}", "w") as f:
    dt = np.dtype("S10")
    f.create_dataset("names", data=np.array([b"alice", b"bob", b"charlie"], dtype=dt))
"#
    );
    run_python(&script);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("names").unwrap();
    let strings = ds.read_string().unwrap();
    assert_eq!(strings, vec!["alice", "bob", "charlie"]);
}

// ---------------------------------------------------------------------------
// B8. h5py creates chunked+compressed+shuffled -> purehdf5 reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn h5py_writes_chunked_compressed_purehdf5_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5py_chunked.h5");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import h5py, numpy as np
data = np.arange(2000, dtype=np.float64) * 0.1
with h5py.File("{path_str}", "w") as f:
    f.create_dataset("compressed", data=data, chunks=(200,),
                     compression="gzip", compression_opts=4, shuffle=True)
"#
    );
    run_python(&script);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("compressed").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.shape().unwrap(), vec![2000]);
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), 2000);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1999] - 199.9).abs() < 1e-10);
    assert!((values[1000] - 100.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// B9. h5py creates nested groups with attrs -> purehdf5 reads -> verify
// ---------------------------------------------------------------------------

#[test]
fn h5py_writes_nested_groups_purehdf5_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5py_groups.h5");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File("{path_str}", "w") as f:
    f.attrs["file_attr"] = "root_value"
    g1 = f.create_group("level1")
    g1.attrs["g1_attr"] = 42
    g2 = g1.create_group("level2")
    g2.attrs["g2_attr"] = 3.25
    g2.create_dataset("deep_data", data=np.array([100.0, 200.0], dtype=np.float64))
"#
    );
    run_python(&script);

    let file = File::open(&path).unwrap();

    // Root attrs
    let root_attrs = file.root().attrs().unwrap();
    assert!(
        matches!(root_attrs.get("file_attr"), Some(AttrValue::String(s)) if s == "root_value")
    );

    // Level1 group
    let g1 = file.group("level1").unwrap();
    let g1_attrs = g1.attrs().unwrap();
    assert!(matches!(g1_attrs.get("g1_attr"), Some(AttrValue::I64(42))));

    // Level2 group (nested)
    let g2 = file.group("level1/level2").unwrap();
    let g2_attrs = g2.attrs().unwrap();
    assert!(
        matches!(g2_attrs.get("g2_attr"), Some(AttrValue::F64(v)) if (*v - 3.25).abs() < 1e-10)
    );

    // Deep dataset
    let ds = file.dataset("level1/level2/deep_data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![100.0, 200.0]);
}

// ---------------------------------------------------------------------------
// B10. h5py creates multiple dtypes -> purehdf5 reads all -> verify
// ---------------------------------------------------------------------------

#[test]
fn h5py_writes_multiple_dtypes_purehdf5_reads() {
    skip_if_no_python!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5py_dtypes.h5");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File("{path_str}", "w") as f:
    f.create_dataset("int8", data=np.array([1, -1, 127], dtype=np.int8))
    f.create_dataset("int16", data=np.array([256, -256], dtype=np.int16))
    f.create_dataset("int32", data=np.array([100000, -100000], dtype=np.int32))
    f.create_dataset("int64", data=np.array([2**40, -2**40], dtype=np.int64))
    f.create_dataset("float32", data=np.array([1.5, 2.5], dtype=np.float32))
    f.create_dataset("float64", data=np.array([3.25, 2.75], dtype=np.float64))
    f.create_dataset("uint8", data=np.array([0, 128, 255], dtype=np.uint8))
"#
    );
    run_python(&script);

    let file = File::open(&path).unwrap();

    // int8 -> read as i32 (upcast)
    let ds = file.dataset("int8").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I8);
    let vals = ds.read_i32().unwrap();
    assert_eq!(vals, vec![1, -1, 127]);

    // int16
    let ds = file.dataset("int16").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I16);
    let vals = ds.read_i32().unwrap();
    assert_eq!(vals, vec![256, -256]);

    // int32
    let ds = file.dataset("int32").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I32);
    assert_eq!(ds.read_i32().unwrap(), vec![100000, -100000]);

    // int64
    let ds = file.dataset("int64").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I64);
    assert_eq!(ds.read_i64().unwrap(), vec![1i64 << 40, -(1i64 << 40)]);

    // float32
    let ds = file.dataset("float32").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F32);
    assert_eq!(ds.read_f32().unwrap(), vec![1.5f32, 2.5]);

    // float64
    let ds = file.dataset("float64").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    let vals = ds.read_f64().unwrap();
    assert!((vals[0] - 3.25).abs() < 1e-10);
    assert!((vals[1] - 2.75).abs() < 1e-10);

    // uint8
    let ds = file.dataset("uint8").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U8);
    let vals = ds.read_u64().unwrap();
    assert_eq!(vals, vec![0, 128, 255]);
}

// ---------------------------------------------------------------------------
// Minimal JSON parsing (avoid serde dependency)
// ---------------------------------------------------------------------------

fn serde_json_minimal_parse(s: &str) -> Vec<f64> {
    // Parse a JSON array of numbers like "[1.5, 2.5, 3.5]"
    let s = s.trim();
    let s = s.strip_prefix('[').unwrap_or(s);
    let s = s.strip_suffix(']').unwrap_or(s);
    s.split(',')
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect()
}
