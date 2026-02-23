//! PyAttrs — dict-like access to HDF5 attributes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::{attr_value_to_py, py_to_attr_value, OwnedAttrValue};

/// Backing storage for attributes.
enum AttrsInner {
    /// Read-only attributes from an existing HDF5 object.
    Read(HashMap<String, rustyhdf5_rs::AttrValue>),
    /// Writable attribute list shared with a parent (PyFile or PyGroup).
    Write(Arc<Mutex<Vec<(String, OwnedAttrValue)>>>),
}

/// Dict-like access to HDF5 attributes.
///
/// In read mode, provides immutable access to attribute key/value pairs.
/// In write mode, attributes set here are accumulated and written when
/// the parent file is closed.
#[pyclass(name = "Attrs")]
pub struct PyAttrs {
    inner: AttrsInner,
}

impl PyAttrs {
    /// Create a read-only attrs from an existing attribute map.
    pub(crate) fn from_read(map: HashMap<String, rustyhdf5_rs::AttrValue>) -> Self {
        Self {
            inner: AttrsInner::Read(map),
        }
    }

    /// Create a writable attrs that shares storage with a parent object.
    pub(crate) fn from_write(store: Arc<Mutex<Vec<(String, OwnedAttrValue)>>>) -> Self {
        Self {
            inner: AttrsInner::Write(store),
        }
    }
}

#[pymethods]
impl PyAttrs {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match &self.inner {
            AttrsInner::Read(map) => match map.get(key) {
                Some(val) => Ok(attr_value_to_py(py, val)),
                None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    key.to_string(),
                )),
            },
            AttrsInner::Write(store) => {
                let guard = store.lock().unwrap();
                for (k, v) in guard.iter() {
                    if k == key {
                        let attr_val: rustyhdf5_rs::AttrValue = v.clone().into();
                        return Ok(attr_value_to_py(py, &attr_val));
                    }
                }
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    key.to_string(),
                ))
            }
        }
    }

    fn __setitem__(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match &self.inner {
            AttrsInner::Read(_) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "cannot set attributes on a read-only file",
            )),
            AttrsInner::Write(store) => {
                let owned = py_to_attr_value(value)?;
                let mut guard = store.lock().unwrap();
                // Replace existing key if present.
                if let Some(entry) = guard.iter_mut().find(|(k, _)| k == key) {
                    entry.1 = owned;
                } else {
                    guard.push((key.to_string(), owned));
                }
                Ok(())
            }
        }
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            AttrsInner::Read(map) => map.len(),
            AttrsInner::Write(store) => store.lock().unwrap().len(),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        match &self.inner {
            AttrsInner::Read(map) => map.contains_key(key),
            AttrsInner::Write(store) => store.lock().unwrap().iter().any(|(k, _)| k == key),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let keys = self.keys(py)?;
        let iter = keys.call_method0(py, "__iter__")?;
        Ok(iter)
    }

    fn __repr__(&self) -> String {
        let n = self.__len__();
        format!("<HDF5 Attrs ({n} members)>")
    }

    /// Return attribute names as a list.
    fn keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        let names: Vec<String> = match &self.inner {
            AttrsInner::Read(map) => map.keys().cloned().collect(),
            AttrsInner::Write(store) => {
                store.lock().unwrap().iter().map(|(k, _)| k.clone()).collect()
            }
        };
        let list = PyList::new(py, &names)?;
        Ok(list.into_any().unbind())
    }

    /// Return attribute values as a list.
    fn values(&self, py: Python<'_>) -> PyResult<PyObject> {
        let vals: Vec<PyObject> = match &self.inner {
            AttrsInner::Read(map) => map.values().map(|v| attr_value_to_py(py, v)).collect(),
            AttrsInner::Write(store) => store
                .lock()
                .unwrap()
                .iter()
                .map(|(_, v)| {
                    let attr: rustyhdf5_rs::AttrValue = v.clone().into();
                    attr_value_to_py(py, &attr)
                })
                .collect(),
        };
        let list = PyList::new(py, &vals)?;
        Ok(list.into_any().unbind())
    }

    /// Return attribute (key, value) pairs as a list of tuples.
    fn items(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pairs: Vec<(String, PyObject)> = match &self.inner {
            AttrsInner::Read(map) => map
                .iter()
                .map(|(k, v)| (k.clone(), attr_value_to_py(py, v)))
                .collect(),
            AttrsInner::Write(store) => store
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    let attr: rustyhdf5_rs::AttrValue = v.clone().into();
                    (k.clone(), attr_value_to_py(py, &attr))
                })
                .collect(),
        };
        let list = PyList::new(py, &pairs)?;
        Ok(list.into_any().unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_attrs_len() {
        let mut map = HashMap::new();
        map.insert("a".into(), rustyhdf5_rs::AttrValue::I64(1));
        map.insert("b".into(), rustyhdf5_rs::AttrValue::F64(2.0));
        let attrs = PyAttrs::from_read(map);
        assert_eq!(attrs.__len__(), 2);
    }

    #[test]
    fn read_attrs_contains() {
        let mut map = HashMap::new();
        map.insert("x".into(), rustyhdf5_rs::AttrValue::String("hello".into()));
        let attrs = PyAttrs::from_read(map);
        assert!(attrs.__contains__("x"));
        assert!(!attrs.__contains__("y"));
    }

    #[test]
    fn write_attrs_len() {
        let store = Arc::new(Mutex::new(Vec::new()));
        store
            .lock()
            .unwrap()
            .push(("key".into(), OwnedAttrValue::I64(99)));
        let attrs = PyAttrs::from_write(store);
        assert_eq!(attrs.__len__(), 1);
    }
}
