//! PyFile — the main entry point for opening and creating HDF5 files.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::attrs::PyAttrs;
use crate::dataset::PyDataset;
use crate::group::{finalize_write_group, PyGroup, WriteGroupState};
use crate::{
    apply_dataset_spec, extract_numpy_data, to_py_err, DatasetSpec, OwnedAttrValue,
};

/// Internal state for write mode.
struct WriteState {
    path: PathBuf,
    root_datasets: Vec<DatasetSpec>,
    root_attrs: Arc<Mutex<Vec<(String, OwnedAttrValue)>>>,
    groups: Vec<Arc<Mutex<WriteGroupState>>>,
}

/// An open HDF5 file.
///
/// Mirrors the h5py.File interface:
///
/// ```python
/// # Reading
/// f = rustyhdf5.File('data.h5', 'r')
/// ds = f['dataset']
/// f.close()
///
/// # Writing
/// with rustyhdf5.File('out.h5', 'w') as f:
///     f.create_dataset('data', data=numpy_array)
/// ```
#[pyclass(name = "File")]
pub struct PyFile {
    inner: Option<FileInner>,
}

enum FileInner {
    Read(Arc<rustyhdf5_rs::File>),
    Write(WriteState),
}

#[pymethods]
impl PyFile {
    /// Open or create an HDF5 file.
    ///
    /// Parameters:
    ///   path: file path
    ///   mode: 'r' for read (default), 'w' for write
    #[new]
    #[pyo3(signature = (path, mode="r"))]
    fn new(path: &str, mode: &str) -> PyResult<Self> {
        match mode {
            "r" => {
                let file = rustyhdf5_rs::File::open(path).map_err(to_py_err)?;
                Ok(Self {
                    inner: Some(FileInner::Read(Arc::new(file))),
                })
            }
            "w" => Ok(Self {
                inner: Some(FileInner::Write(WriteState {
                    path: PathBuf::from(path),
                    root_datasets: Vec::new(),
                    root_attrs: Arc::new(Mutex::new(Vec::new())),
                    groups: Vec::new(),
                })),
            }),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported mode '{other}'; expected 'r' or 'w'"
            ))),
        }
    }

    /// Close the file. In write mode, this finalizes and writes the file.
    fn close(&mut self) -> PyResult<()> {
        let inner = self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>("file is already closed")
        })?;
        match inner {
            FileInner::Read(_) => Ok(()),
            FileInner::Write(state) => finalize_write(state),
        }
    }

    /// Context manager entry — returns self.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit — closes the file.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false) // don't suppress exceptions
    }

    /// Get a child object (dataset or group) by path.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let file = self.read_file()?;
        // Try dataset first
        match file.dataset(key) {
            Ok(_) => {
                let ds = PyDataset::new(Arc::clone(file), key.to_string())?;
                Ok(ds.into_pyobject(py)?.into_any().unbind())
            }
            Err(rustyhdf5_rs::Error::NotADataset(_)) => {
                let grp = PyGroup::from_read(Arc::clone(file), key.to_string());
                Ok(grp.into_pyobject(py)?.into_any().unbind())
            }
            Err(_) => {
                // Could be a group (no DataLayout message, no error)
                match file.group(key) {
                    Ok(_) => {
                        let grp = PyGroup::from_read(Arc::clone(file), key.to_string());
                        Ok(grp.into_pyobject(py)?.into_any().unbind())
                    }
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        format!("{key}: {e}"),
                    )),
                }
            }
        }
    }

    /// List the names of all children in the root group.
    fn keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        let file = self.read_file()?;
        let root = file.root();
        let mut names = root.datasets().map_err(to_py_err)?;
        let groups = root.groups().map_err(to_py_err)?;
        names.extend(groups);
        names.sort();
        let list = pyo3::types::PyList::new(py, &names)?;
        Ok(list.into_any().unbind())
    }

    /// Create a dataset in the root group (write mode only).
    ///
    /// Parameters:
    ///   name: dataset name
    ///   data: numpy array
    ///   chunks: optional chunk dimensions (tuple or list)
    ///   compression: optional, only 'gzip' supported
    ///   compression_opts: gzip compression level (1-9)
    #[pyo3(signature = (name, *, data, chunks=None, compression=None, compression_opts=None))]
    fn create_dataset(
        &mut self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, PyAny>,
        chunks: Option<Vec<u64>>,
        compression: Option<&str>,
        compression_opts: Option<u32>,
    ) -> PyResult<()> {
        let state = self.write_state_mut()?;
        let (dataset_data, shape) = extract_numpy_data(py, data)?;
        let deflate_level = parse_compression(compression, compression_opts)?;
        let spec = DatasetSpec {
            name: name.to_string(),
            data: dataset_data,
            shape,
            chunks,
            deflate_level,
            attrs: vec![],
        };
        state.root_datasets.push(spec);
        Ok(())
    }

    /// Create a group (write mode only). Returns a `Group` handle.
    fn create_group(&mut self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let state = self.write_state_mut()?;
        let group_state = Arc::new(Mutex::new(WriteGroupState {
            name: name.to_string(),
            datasets: vec![],
            attrs: Arc::new(Mutex::new(vec![])),
        }));
        state.groups.push(Arc::clone(&group_state));
        let grp = PyGroup::from_write(group_state);
        Ok(grp.into_pyobject(py)?.into_any().unbind())
    }

    /// Attribute access. In read mode, returns attributes of the root group.
    /// In write mode, returns a writable attrs handle.
    #[getter]
    fn attrs(&self) -> PyResult<PyAttrs> {
        match self.inner.as_ref() {
            Some(FileInner::Read(file)) => {
                let map = file.root().attrs().map_err(to_py_err)?;
                Ok(PyAttrs::from_read(map))
            }
            Some(FileInner::Write(state)) => {
                Ok(PyAttrs::from_write(Arc::clone(&state.root_attrs)))
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "file is closed",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(FileInner::Read(f)) => {
                format!("<HDF5 File (read, {} bytes)>", f.as_bytes().len())
            }
            Some(FileInner::Write(s)) => {
                format!("<HDF5 File (write, \"{}\")>", s.path.display())
            }
            None => "<HDF5 File (closed)>".to_string(),
        }
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        let file = self.read_file()?;
        Ok(file.dataset(key).is_ok() || file.group(key).is_ok())
    }
}

impl PyFile {
    fn read_file(&self) -> PyResult<&Arc<rustyhdf5_rs::File>> {
        match &self.inner {
            Some(FileInner::Read(f)) => Ok(f),
            Some(FileInner::Write(_)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "cannot read from a file opened for writing",
            )),
            None => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "file is closed",
            )),
        }
    }

    fn write_state_mut(&mut self) -> PyResult<&mut WriteState> {
        match &mut self.inner {
            Some(FileInner::Write(s)) => Ok(s),
            Some(FileInner::Read(_)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "cannot write to a file opened for reading",
            )),
            None => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "file is closed",
            )),
        }
    }
}

fn parse_compression(
    compression: Option<&str>,
    compression_opts: Option<u32>,
) -> PyResult<Option<u32>> {
    match compression {
        Some("gzip") => Ok(Some(compression_opts.unwrap_or(4))),
        Some(other) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported compression: {other}; only 'gzip' is supported"
        ))),
        None => Ok(None),
    }
}

/// Build and write the HDF5 file from accumulated write state.
fn finalize_write(state: WriteState) -> PyResult<()> {
    let mut builder = rustyhdf5_rs::FileBuilder::new();

    // Root attributes
    let root_attrs = state.root_attrs.lock().unwrap();
    for (name, val) in root_attrs.iter() {
        builder.set_attr(name, val.clone().into());
    }
    drop(root_attrs);

    // Root datasets
    for spec in &state.root_datasets {
        let db = builder.create_dataset(&spec.name);
        apply_dataset_spec(db, spec);
    }

    // Groups
    for group_arc in &state.groups {
        let guard = group_arc.lock().unwrap();
        finalize_write_group(&mut builder, &guard);
    }

    builder.write(&state.path).map_err(to_py_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gzip_compression() {
        assert_eq!(parse_compression(Some("gzip"), Some(6)).unwrap(), Some(6));
        assert_eq!(parse_compression(Some("gzip"), None).unwrap(), Some(4));
        assert_eq!(parse_compression(None, None).unwrap(), None);
        assert!(parse_compression(Some("lz4"), None).is_err());
    }

    #[test]
    fn finalize_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_py_test_finalize.h5");

        let state = WriteState {
            path: path.clone(),
            root_datasets: vec![DatasetSpec {
                name: "data".into(),
                data: crate::DatasetData::F64(vec![1.0, 2.0, 3.0]),
                shape: vec![3],
                chunks: None,
                deflate_level: None,
                attrs: vec![("unit".into(), OwnedAttrValue::Str("m".into()))],
            }],
            root_attrs: Arc::new(Mutex::new(vec![
                ("version".into(), OwnedAttrValue::I64(1)),
            ])),
            groups: vec![Arc::new(Mutex::new(WriteGroupState {
                name: "grp".into(),
                datasets: vec![DatasetSpec {
                    name: "vals".into(),
                    data: crate::DatasetData::I32(vec![10, 20]),
                    shape: vec![2],
                    chunks: None,
                    deflate_level: None,
                    attrs: vec![],
                }],
                attrs: Arc::new(Mutex::new(vec![])),
            }))],
        };

        finalize_write(state).unwrap();

        let file = rustyhdf5_rs::File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
        let grp_ds = file.dataset("grp/vals").unwrap();
        assert_eq!(grp_ds.read_i32().unwrap(), vec![10, 20]);

        std::fs::remove_file(&path).ok();
    }
}
