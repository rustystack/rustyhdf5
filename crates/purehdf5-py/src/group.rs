//! PyGroup — navigable HDF5 group with read and write support.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::attrs::PyAttrs;
use crate::dataset::PyDataset;
use crate::{
    apply_dataset_spec, extract_numpy_data, to_py_err, DatasetSpec, OwnedAttrValue,
};

/// Shared state for a group being written.
pub(crate) struct WriteGroupState {
    pub name: String,
    pub datasets: Vec<DatasetSpec>,
    pub attrs: Arc<Mutex<Vec<(String, OwnedAttrValue)>>>,
}

/// An HDF5 group.
///
/// In read mode, provides `__getitem__` navigation and child listing.
/// In write mode, supports `create_dataset` and `create_group` and
/// attribute setting.
///
/// ```python
/// grp = f['group_name']
/// grp.keys()
/// ds = grp['dataset']
/// ```
#[pyclass(name = "Group")]
pub struct PyGroup {
    inner: GroupInner,
}

enum GroupInner {
    Read {
        file: Arc<purehdf5_rs::File>,
        path: String,
    },
    Write(Arc<Mutex<WriteGroupState>>),
}

impl PyGroup {
    pub(crate) fn from_read(file: Arc<purehdf5_rs::File>, path: String) -> Self {
        Self {
            inner: GroupInner::Read { file, path },
        }
    }

    pub(crate) fn from_write(state: Arc<Mutex<WriteGroupState>>) -> Self {
        Self {
            inner: GroupInner::Write(state),
        }
    }
}

#[pymethods]
impl PyGroup {
    /// Get a child object (dataset or subgroup) by name or path.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match &self.inner {
            GroupInner::Read { file, path } => {
                let full_path = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}/{key}")
                };
                // Try dataset first
                match file.dataset(&full_path) {
                    Ok(_) => {
                        let ds = PyDataset::new(Arc::clone(file), full_path)?;
                        Ok(ds.into_pyobject(py)?.into_any().unbind())
                    }
                    Err(purehdf5_rs::Error::NotADataset(_)) => {
                        let grp = PyGroup::from_read(Arc::clone(file), full_path);
                        Ok(grp.into_pyobject(py)?.into_any().unbind())
                    }
                    Err(e) => {
                        // Could be a group without a DataLayout message
                        match file.group(&full_path) {
                            Ok(_) => {
                                let grp =
                                    PyGroup::from_read(Arc::clone(file), full_path);
                                Ok(grp.into_pyobject(py)?.into_any().unbind())
                            }
                            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                                format!("{key}: {e}"),
                            )),
                        }
                    }
                }
            }
            GroupInner::Write(_) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "cannot read children from a group opened for writing",
            )),
        }
    }

    /// List the names of all children (datasets and subgroups).
    fn keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            GroupInner::Read { file, path } => {
                let group = if path.is_empty() {
                    file.root()
                } else {
                    file.group(path).map_err(to_py_err)?
                };
                let mut names = group.datasets().map_err(to_py_err)?;
                let groups = group.groups().map_err(to_py_err)?;
                names.extend(groups);
                names.sort();
                let list = PyList::new(py, &names)?;
                Ok(list.into_any().unbind())
            }
            GroupInner::Write(state) => {
                let guard = state.lock().unwrap();
                let names: Vec<&str> = guard.datasets.iter().map(|d| d.name.as_str()).collect();
                let list = PyList::new(py, &names)?;
                Ok(list.into_any().unbind())
            }
        }
    }

    /// Create a dataset inside this group (write mode only).
    ///
    /// Parameters:
    ///   name: dataset name
    ///   data: numpy array
    ///   chunks: optional chunk dimensions
    ///   compression: optional, only 'gzip' supported
    ///   compression_opts: gzip level (1-9)
    #[pyo3(signature = (name, *, data, chunks=None, compression=None, compression_opts=None))]
    fn create_dataset(
        &self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, PyAny>,
        chunks: Option<Vec<u64>>,
        compression: Option<&str>,
        compression_opts: Option<u32>,
    ) -> PyResult<()> {
        match &self.inner {
            GroupInner::Write(state) => {
                let (dataset_data, shape) = extract_numpy_data(py, data)?;
                let deflate_level = match compression {
                    Some("gzip") => Some(compression_opts.unwrap_or(4)),
                    Some(other) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("unsupported compression: {other}; only 'gzip' is supported"),
                        ));
                    }
                    None => None,
                };
                let spec = DatasetSpec {
                    name: name.to_string(),
                    data: dataset_data,
                    shape,
                    chunks,
                    deflate_level,
                    attrs: vec![],
                };
                state.lock().unwrap().datasets.push(spec);
                Ok(())
            }
            GroupInner::Read { .. } => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "cannot create datasets on a read-only group",
            )),
        }
    }

    /// Attribute access.
    #[getter]
    fn attrs(&self) -> PyResult<PyAttrs> {
        match &self.inner {
            GroupInner::Read { file, path } => {
                let group = if path.is_empty() {
                    file.root()
                } else {
                    file.group(path).map_err(to_py_err)?
                };
                let map = group.attrs().map_err(to_py_err)?;
                Ok(PyAttrs::from_read(map))
            }
            GroupInner::Write(state) => {
                let store = Arc::clone(&state.lock().unwrap().attrs);
                Ok(PyAttrs::from_write(store))
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            GroupInner::Read { path, .. } => {
                if path.is_empty() {
                    "<HDF5 Group \"/\" (root)>".to_string()
                } else {
                    format!("<HDF5 Group \"/{path}\">")
                }
            }
            GroupInner::Write(state) => {
                let name = &state.lock().unwrap().name;
                format!("<HDF5 Group \"{name}\" (write)>")
            }
        }
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        match &self.inner {
            GroupInner::Read { file, path } => {
                let full_path = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}/{key}")
                };
                Ok(file.dataset(&full_path).is_ok() || file.group(&full_path).is_ok())
            }
            GroupInner::Write(state) => {
                let guard = state.lock().unwrap();
                Ok(guard.datasets.iter().any(|d| d.name == key))
            }
        }
    }
}

/// Finalize a write group into the file builder.
pub(crate) fn finalize_write_group(
    builder: &mut purehdf5_rs::FileBuilder,
    state: &WriteGroupState,
) {
    let mut gb = builder.create_group(&state.name);
    for spec in &state.datasets {
        let db = gb.create_dataset(&spec.name);
        apply_dataset_spec(db, spec);
    }
    let attrs_guard = state.attrs.lock().unwrap();
    for (name, val) in attrs_guard.iter() {
        gb.set_attr(name, val.clone().into());
    }
    let finished = gb.finish();
    builder.add_group(finished);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_group_construction() {
        let mut b = purehdf5_rs::FileBuilder::new();
        let mut g = b.create_group("grp");
        g.create_dataset("x").with_f64_data(&[1.0]);
        let finished = g.finish();
        b.add_group(finished);
        let bytes = b.finish().unwrap();
        let file = Arc::new(purehdf5_rs::File::from_bytes(bytes).unwrap());
        let _grp = PyGroup::from_read(file, "grp".into());
    }

    #[test]
    fn write_group_state() {
        let state = WriteGroupState {
            name: "test".into(),
            datasets: vec![],
            attrs: Arc::new(Mutex::new(vec![])),
        };
        let arc = Arc::new(Mutex::new(state));
        let _grp = PyGroup::from_write(arc);
    }

    #[test]
    fn finalize_group() {
        let state = WriteGroupState {
            name: "mygroup".into(),
            datasets: vec![DatasetSpec {
                name: "vals".into(),
                data: crate::DatasetData::F64(vec![1.0, 2.0]),
                shape: vec![2],
                chunks: None,
                deflate_level: None,
                attrs: vec![],
            }],
            attrs: Arc::new(Mutex::new(vec![
                ("version".into(), OwnedAttrValue::I64(1)),
            ])),
        };
        let mut builder = purehdf5_rs::FileBuilder::new();
        // Need a root dataset for a valid file
        builder.create_dataset("root_ds").with_f64_data(&[0.0]);
        finalize_write_group(&mut builder, &state);
        let bytes = builder.finish().unwrap();
        let file = purehdf5_rs::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("mygroup/vals").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0]);
    }
}
