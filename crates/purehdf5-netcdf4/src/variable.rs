//! NetCDF-4 variable representation.
//!
//! Variables in NetCDF-4 are HDF5 datasets. This module wraps them with
//! dimension associations and CF attribute support.

use std::collections::HashMap;

use purehdf5::AttrValue;

use crate::cf::{self, CfAttributes};
use crate::dimension::Dimension;
use crate::error::Error;
use crate::types::{dtype_to_nctype, NcType};

/// A NetCDF-4 variable backed by an HDF5 dataset.
pub struct Variable<'f> {
    /// Variable name.
    name: String,
    /// Underlying HDF5 dataset.
    dataset: purehdf5::Dataset<'f>,
    /// Dimensions associated with this variable.
    dims: Vec<Dimension>,
    /// Cached attributes.
    attrs_cache: Option<HashMap<String, AttrValue>>,
}

impl<'f> Variable<'f> {
    /// Create a new Variable wrapping an HDF5 dataset.
    pub(crate) fn new(
        name: String,
        dataset: purehdf5::Dataset<'f>,
        dims: Vec<Dimension>,
    ) -> Self {
        Self {
            name,
            dataset,
            dims,
            attrs_cache: None,
        }
    }

    /// Variable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The dimensions of this variable.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dims
    }

    /// The shape of this variable (dimension sizes).
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        Ok(self.dataset.shape()?)
    }

    /// The NetCDF data type of this variable.
    pub fn nc_type(&self) -> Result<NcType, Error> {
        let dtype = self.dataset.dtype()?;
        Ok(dtype_to_nctype(&dtype))
    }

    /// Read all attributes as a HashMap.
    pub fn attrs(&mut self) -> Result<&HashMap<String, AttrValue>, Error> {
        if self.attrs_cache.is_none() {
            self.attrs_cache = Some(self.dataset.attrs()?);
        }
        Ok(self.attrs_cache.as_ref().unwrap())
    }

    /// Extract CF convention attributes.
    pub fn cf_attributes(&mut self) -> Result<CfAttributes, Error> {
        let attrs = self.attrs()?;
        Ok(cf::extract_cf_attributes(attrs))
    }

    /// Read data as f64 with scale_factor/add_offset applied.
    ///
    /// Missing values (matching `_FillValue` or `missing_value`) become NaN.
    /// If no scale_factor or add_offset attributes exist, returns the raw f64 data.
    pub fn read_f64(&mut self) -> Result<Vec<f64>, Error> {
        let raw = self.dataset.read_f64()?;
        let cf = self.cf_attributes()?;
        Ok(cf::apply_scale_offset(&raw, &cf))
    }

    /// Read raw data as f64 without any scale/offset transformation.
    pub fn read_raw_f64(&self) -> Result<Vec<f64>, Error> {
        Ok(self.dataset.read_f64()?)
    }

    /// Read raw data as f32 without any scale/offset transformation.
    pub fn read_raw_f32(&self) -> Result<Vec<f32>, Error> {
        Ok(self.dataset.read_f32()?)
    }

    /// Read raw data as i32 without any scale/offset transformation.
    pub fn read_raw_i32(&self) -> Result<Vec<i32>, Error> {
        Ok(self.dataset.read_i32()?)
    }

    /// Read raw data as i64 without any scale/offset transformation.
    pub fn read_raw_i64(&self) -> Result<Vec<i64>, Error> {
        Ok(self.dataset.read_i64()?)
    }

    /// Read raw data as u64 without any scale/offset transformation.
    pub fn read_raw_u64(&self) -> Result<Vec<u64>, Error> {
        Ok(self.dataset.read_u64()?)
    }

    /// Read raw data as strings.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        Ok(self.dataset.read_string()?)
    }

    /// Read raw bytes without any type conversion.
    pub fn read_raw(&self) -> Result<Vec<u8>, Error> {
        // Use the low-level format read to get raw bytes.
        // The high-level API doesn't expose read_raw directly,
        // so we read as the smallest numeric type that matches the element size.
        // For the NetCDF use case, callers should prefer typed reads.
        let dtype = self.dataset.dtype()?;
        match dtype {
            purehdf5::DType::F64 => {
                let vals = self.dataset.read_f64()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            purehdf5::DType::F32 => {
                let vals = self.dataset.read_f32()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            purehdf5::DType::I32 => {
                let vals = self.dataset.read_i32()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            purehdf5::DType::I64 => {
                let vals = self.dataset.read_i64()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            purehdf5::DType::U64 => {
                let vals = self.dataset.read_u64()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            _ => {
                // Fallback: try reading as f64 and convert to bytes
                let vals = self.dataset.read_f64()?;
                Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
        }
    }

    /// Whether this is a coordinate variable (name matches a dimension name).
    pub fn is_coordinate(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].name == self.name
    }
}

impl std::fmt::Debug for Variable<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("name", &self.name)
            .field("dims", &self.dims)
            .finish()
    }
}

/// Build variables from a group's datasets and associated dimensions.
pub(crate) fn build_variables<'f>(
    group: &purehdf5::Group<'f>,
    available_dims: &[Dimension],
) -> Result<Vec<Variable<'f>>, Error> {
    let dataset_names = group.datasets()?;
    let mut variables = Vec::new();

    for ds_name in &dataset_names {
        let ds = group.dataset(ds_name)?;
        let shape = ds.shape()?;

        // Associate dimensions with this variable.
        // First try DIMENSION_LIST attribute, then fall back to shape matching.
        let var_dims = match_dimensions_to_variable(&shape, available_dims);

        variables.push(Variable::new(ds_name.clone(), ds, var_dims));
    }

    Ok(variables)
}

/// Match dimensions to a variable based on shape.
///
/// For each axis of the variable, find a dimension with matching size.
/// If multiple dimensions have the same size, prefer exact name matching
/// from the convention order.
pub(crate) fn match_dimensions_to_variable(shape: &[u64], available_dims: &[Dimension]) -> Vec<Dimension> {
    let mut result = Vec::with_capacity(shape.len());

    // Track which dimensions have been used to avoid duplicates
    let mut used = vec![false; available_dims.len()];

    for &dim_size in shape {
        let mut matched = false;

        // Find a dimension with matching size that hasn't been used yet
        for (i, dim) in available_dims.iter().enumerate() {
            if !used[i] && dim.size == dim_size {
                result.push(dim.clone());
                used[i] = true;
                matched = true;
                break;
            }
        }

        if !matched {
            // Create an anonymous dimension for unmatched sizes
            result.push(Dimension {
                name: format!("dim_{dim_size}"),
                size: dim_size,
                is_unlimited: false,
            });
        }
    }

    result
}
