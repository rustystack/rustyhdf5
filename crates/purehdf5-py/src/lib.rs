//! Python bindings for purehdf5 — a pure-Rust HDF5 library.
//!
//! Provides a Pythonic API mirroring h5py:
//!
//! ```python
//! import purehdf5
//!
//! with purehdf5.File('data.h5', 'r') as f:
//!     data = f['dataset_name'][:]
//! ```

mod attrs;
mod dataset;
mod file;
mod group;

use pyo3::prelude::*;

pub(crate) use attrs::PyAttrs;
pub(crate) use dataset::PyDataset;
pub(crate) use file::PyFile;
pub(crate) use group::PyGroup;

/// Convert a `purehdf5::Error` into a `PyErr`.
pub(crate) fn to_py_err(e: purehdf5_rs::Error) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string())
}

/// The data payload for a dataset being written.
#[derive(Clone)]
pub(crate) enum DatasetData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    U8(Vec<u8>),
}

/// Specification for a dataset to be written.
#[derive(Clone)]
pub(crate) struct DatasetSpec {
    pub name: String,
    pub data: DatasetData,
    pub shape: Vec<u64>,
    pub chunks: Option<Vec<u64>>,
    pub deflate_level: Option<u32>,
    pub attrs: Vec<(String, OwnedAttrValue)>,
}

/// Owned attribute value (used during write accumulation).
#[derive(Clone)]
pub(crate) enum OwnedAttrValue {
    F64(f64),
    I64(i64),
    Str(String),
    F64Array(Vec<f64>),
    I64Array(Vec<i64>),
}

impl From<OwnedAttrValue> for purehdf5_rs::AttrValue {
    fn from(v: OwnedAttrValue) -> Self {
        match v {
            OwnedAttrValue::F64(x) => purehdf5_rs::AttrValue::F64(x),
            OwnedAttrValue::I64(x) => purehdf5_rs::AttrValue::I64(x),
            OwnedAttrValue::Str(s) => purehdf5_rs::AttrValue::String(s),
            OwnedAttrValue::F64Array(a) => purehdf5_rs::AttrValue::F64Array(a),
            OwnedAttrValue::I64Array(a) => purehdf5_rs::AttrValue::I64Array(a),
        }
    }
}

/// Extract a Python value into an `OwnedAttrValue`.
pub(crate) fn py_to_attr_value(val: &Bound<'_, PyAny>) -> PyResult<OwnedAttrValue> {
    // Try int first (before float, since bool is int subclass in Python)
    if let Ok(v) = val.extract::<i64>() {
        return Ok(OwnedAttrValue::I64(v));
    }
    if let Ok(v) = val.extract::<f64>() {
        return Ok(OwnedAttrValue::F64(v));
    }
    if let Ok(v) = val.extract::<String>() {
        return Ok(OwnedAttrValue::Str(v));
    }
    // Try list of floats, list of ints
    if let Ok(v) = val.extract::<Vec<f64>>() {
        return Ok(OwnedAttrValue::F64Array(v));
    }
    if let Ok(v) = val.extract::<Vec<i64>>() {
        return Ok(OwnedAttrValue::I64Array(v));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "unsupported attribute type; expected int, float, str, or list of int/float",
    ))
}

/// Convert an `AttrValue` (from the Rust lib) to a Python object.
pub(crate) fn attr_value_to_py(py: Python<'_>, val: &purehdf5_rs::AttrValue) -> PyObject {
    match val {
        purehdf5_rs::AttrValue::F64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        purehdf5_rs::AttrValue::I64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        purehdf5_rs::AttrValue::U64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        purehdf5_rs::AttrValue::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        purehdf5_rs::AttrValue::F64Array(a) => {
            let list = pyo3::types::PyList::new(py, a).unwrap();
            list.into_any().unbind()
        }
        purehdf5_rs::AttrValue::I64Array(a) => {
            let list = pyo3::types::PyList::new(py, a).unwrap();
            list.into_any().unbind()
        }
        purehdf5_rs::AttrValue::StringArray(a) => {
            let list = pyo3::types::PyList::new(py, a).unwrap();
            list.into_any().unbind()
        }
    }
}

/// Apply a `DatasetSpec` to a `DatasetBuilder`.
pub(crate) fn apply_dataset_spec(
    db: &mut purehdf5_format::type_builders::DatasetBuilder,
    spec: &DatasetSpec,
) {
    match &spec.data {
        DatasetData::F64(v) => {
            db.with_f64_data(v);
        }
        DatasetData::F32(v) => {
            db.with_f32_data(v);
        }
        DatasetData::I64(v) => {
            db.with_i64_data(v);
        }
        DatasetData::I32(v) => {
            db.with_i32_data(v);
        }
        DatasetData::U8(v) => {
            db.with_u8_data(v);
        }
    }
    if !spec.shape.is_empty() {
        db.with_shape(&spec.shape);
    }
    if let Some(chunks) = &spec.chunks {
        db.with_chunks(chunks);
    }
    if let Some(level) = spec.deflate_level {
        db.with_deflate(level);
    }
    for (name, val) in &spec.attrs {
        db.set_attr(name, val.clone().into());
    }
}

/// Extract numpy array data from a Python object.
pub(crate) fn extract_numpy_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
) -> PyResult<(DatasetData, Vec<u64>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data,))?;
    let dtype_str: String = arr.getattr("dtype")?.str()?.extract()?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
    let flat = arr.call_method0("ravel")?;

    let dataset_data = match dtype_str.as_str() {
        "float64" => DatasetData::F64(flat.extract::<Vec<f64>>()?),
        "float32" => DatasetData::F32(flat.extract::<Vec<f32>>()?),
        "int64" => DatasetData::I64(flat.extract::<Vec<i64>>()?),
        "int32" => DatasetData::I32(flat.extract::<Vec<i32>>()?),
        "uint8" => DatasetData::U8(flat.extract::<Vec<u8>>()?),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported numpy dtype: {dtype_str}; expected float64, float32, int64, int32, or uint8"
            )));
        }
    };
    Ok((dataset_data, shape_u64))
}

/// The purehdf5 Python module.
#[pymodule]
fn purehdf5(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFile>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyAttrs>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_attr_value_roundtrip() {
        let val = OwnedAttrValue::F64(3.14);
        let attr: purehdf5_rs::AttrValue = val.into();
        assert!(matches!(attr, purehdf5_rs::AttrValue::F64(v) if (v - 3.14).abs() < 1e-10));
    }

    #[test]
    fn owned_attr_value_i64() {
        let val = OwnedAttrValue::I64(42);
        let attr: purehdf5_rs::AttrValue = val.into();
        assert!(matches!(attr, purehdf5_rs::AttrValue::I64(42)));
    }

    #[test]
    fn owned_attr_value_str() {
        let val = OwnedAttrValue::Str("hello".into());
        let attr: purehdf5_rs::AttrValue = val.into();
        assert!(matches!(attr, purehdf5_rs::AttrValue::String(ref s) if s == "hello"));
    }

    #[test]
    fn owned_attr_value_f64_array() {
        let val = OwnedAttrValue::F64Array(vec![1.0, 2.0]);
        let attr: purehdf5_rs::AttrValue = val.into();
        assert!(matches!(attr, purehdf5_rs::AttrValue::F64Array(ref v) if v == &[1.0, 2.0]));
    }

    #[test]
    fn owned_attr_value_i64_array() {
        let val = OwnedAttrValue::I64Array(vec![1, 2, 3]);
        let attr: purehdf5_rs::AttrValue = val.into();
        assert!(matches!(attr, purehdf5_rs::AttrValue::I64Array(ref v) if v == &[1, 2, 3]));
    }

    #[test]
    fn dataset_spec_apply() {
        let spec = DatasetSpec {
            name: "test".into(),
            data: DatasetData::F64(vec![1.0, 2.0, 3.0]),
            shape: vec![3],
            chunks: None,
            deflate_level: None,
            attrs: vec![],
        };
        let mut builder = purehdf5_rs::FileBuilder::new();
        let db = builder.create_dataset(&spec.name);
        apply_dataset_spec(db, &spec);
        let bytes = builder.finish().unwrap();
        let file = purehdf5_rs::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("test").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dataset_spec_with_chunks_and_deflate() {
        let spec = DatasetSpec {
            name: "compressed".into(),
            data: DatasetData::I32(vec![10, 20, 30, 40, 50]),
            shape: vec![5],
            chunks: Some(vec![5]),
            deflate_level: Some(4),
            attrs: vec![("unit".into(), OwnedAttrValue::Str("m".into()))],
        };
        let mut builder = purehdf5_rs::FileBuilder::new();
        let db = builder.create_dataset(&spec.name);
        apply_dataset_spec(db, &spec);
        let bytes = builder.finish().unwrap();
        let file = purehdf5_rs::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("compressed").unwrap();
        assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30, 40, 50]);
    }
}
