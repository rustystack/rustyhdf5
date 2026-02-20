//! PyDataset — read access to HDF5 datasets with numpy integration.

use std::sync::Arc;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyList;

use purehdf5_rs::DType;

use crate::attrs::PyAttrs;
use crate::to_py_err;

/// A handle to an HDF5 dataset (read mode).
///
/// Supports numpy-style indexing via `__getitem__`:
/// ```python
/// ds = f['dataset_name']
/// data = ds[:]           # read all data as numpy array
/// shape = ds.shape
/// dtype = ds.dtype
/// ```
#[pyclass(name = "Dataset")]
pub struct PyDataset {
    file: Arc<purehdf5_rs::File>,
    path: String,
    cached_shape: Vec<u64>,
    cached_dtype: DType,
}

impl PyDataset {
    pub fn new(file: Arc<purehdf5_rs::File>, path: String) -> PyResult<Self> {
        let ds = file.dataset(&path).map_err(to_py_err)?;
        let cached_shape = ds.shape().map_err(to_py_err)?;
        let cached_dtype = ds.dtype().map_err(to_py_err)?;
        Ok(Self {
            file,
            path,
            cached_shape,
            cached_dtype,
        })
    }
}

/// Map a `DType` to a numpy dtype string.
fn dtype_to_numpy_str(dt: &DType) -> &'static str {
    match dt {
        DType::F64 => "float64",
        DType::F32 => "float32",
        DType::I64 => "int64",
        DType::I32 => "int32",
        DType::I16 => "int16",
        DType::I8 => "int8",
        DType::U64 => "uint64",
        DType::U32 => "uint32",
        DType::U16 => "uint16",
        DType::U8 => "uint8",
        DType::String | DType::VariableLengthString => "object",
        _ => "object",
    }
}

#[pymethods]
impl PyDataset {
    /// The shape of the dataset as a tuple.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tuple = pyo3::types::PyTuple::new(
            py,
            self.cached_shape.iter().map(|&d| d as usize),
        )?;
        Ok(tuple.into_any().unbind())
    }

    /// The numpy dtype string of the dataset.
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_to_numpy_str(&self.cached_dtype)
    }

    /// Attribute access (read-only).
    #[getter]
    fn attrs(&self) -> PyResult<PyAttrs> {
        let ds = self.file.dataset(&self.path).map_err(to_py_err)?;
        let map = ds.attrs().map_err(to_py_err)?;
        Ok(PyAttrs::from_read(map))
    }

    /// Read data via indexing. Supports `ds[:]`, `ds[0]`, `ds[0:5]`, etc.
    ///
    /// The full dataset is always read from the underlying file; the index
    /// is then applied on the resulting numpy array.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        let arr = self.read_as_numpy(py)?;
        let indexed = arr.get_item(key)?;
        Ok(indexed.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "<HDF5 Dataset \"{}\": shape {:?}, dtype {}>",
            self.path,
            self.cached_shape,
            dtype_to_numpy_str(&self.cached_dtype),
        )
    }

    fn __len__(&self) -> usize {
        self.cached_shape.first().copied().unwrap_or(0) as usize
    }
}

impl PyDataset {
    /// Read the full dataset and return it as a numpy array (or list for strings).
    fn read_as_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.file.dataset(&self.path).map_err(to_py_err)?;
        let shape: Vec<usize> = self.cached_shape.iter().map(|&d| d as usize).collect();

        match &self.cached_dtype {
            DType::F64 => {
                let data = ds.read_f64().map_err(to_py_err)?;
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::F32 => {
                let data = ds.read_f32().map_err(to_py_err)?;
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::I32 => {
                let data = ds.read_i32().map_err(to_py_err)?;
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::I64 => {
                let data = ds.read_i64().map_err(to_py_err)?;
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::U8 => {
                // read_u64 handles u8 data correctly (reads 1-byte values)
                let raw = ds.read_u64().map_err(to_py_err)?;
                let data: Vec<u8> = raw.iter().map(|&v| v as u8).collect();
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::U64 => {
                let data = ds.read_u64().map_err(to_py_err)?;
                let nd = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let arr = PyArrayDyn::from_owned_array(py, nd);
                Ok(arr.into_any())
            }
            DType::String | DType::VariableLengthString => {
                let data = ds.read_string().map_err(to_py_err)?;
                let list = PyList::new(py, &data)?;
                Ok(list.into_any())
            }
            other => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported dataset dtype for reading: {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_mapping() {
        assert_eq!(dtype_to_numpy_str(&DType::F64), "float64");
        assert_eq!(dtype_to_numpy_str(&DType::F32), "float32");
        assert_eq!(dtype_to_numpy_str(&DType::I32), "int32");
        assert_eq!(dtype_to_numpy_str(&DType::I64), "int64");
        assert_eq!(dtype_to_numpy_str(&DType::U8), "uint8");
        assert_eq!(dtype_to_numpy_str(&DType::String), "object");
    }

    #[test]
    fn dataset_from_file() {
        let mut b = purehdf5_rs::FileBuilder::new();
        b.create_dataset("vals").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = b.finish().unwrap();
        let file = Arc::new(purehdf5_rs::File::from_bytes(bytes).unwrap());
        let ds = PyDataset::new(file, "vals".into()).unwrap();
        assert_eq!(ds.cached_shape, vec![3]);
        assert_eq!(ds.cached_dtype, DType::F64);
    }

    #[test]
    fn dataset_len() {
        let mut b = purehdf5_rs::FileBuilder::new();
        b.create_dataset("data")
            .with_i32_data(&[10, 20, 30, 40])
            .with_shape(&[2, 2]);
        let bytes = b.finish().unwrap();
        let file = Arc::new(purehdf5_rs::File::from_bytes(bytes).unwrap());
        let ds = PyDataset::new(file, "data".into()).unwrap();
        assert_eq!(ds.__len__(), 2);
    }
}
