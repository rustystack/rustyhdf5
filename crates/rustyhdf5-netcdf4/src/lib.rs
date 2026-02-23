//! NetCDF-4 read support built on rustyhdf5.
//!
//! NetCDF-4 files are HDF5 files following specific conventions. This crate
//! provides a high-level API for reading NetCDF-4 metadata (dimensions,
//! variables, CF attributes, groups) from HDF5 files written by netCDF4-python,
//! nco, CDO, or any compliant NetCDF-4 writer.
//!
//! # Example
//!
//! ```no_run
//! use rustyhdf5_netcdf4::NetCDF4File;
//!
//! let file = NetCDF4File::open("data.nc").unwrap();
//! for dim in file.dimensions().unwrap() {
//!     println!("{}: {} (unlimited={})", dim.name, dim.size, dim.is_unlimited);
//! }
//! for mut var in file.variables().unwrap() {
//!     println!("{}: {:?}", var.name(), var.shape().unwrap());
//!     let cf = var.cf_attributes().unwrap();
//!     if let Some(units) = &cf.units {
//!         println!("  units: {units}");
//!     }
//! }
//! ```

pub mod cf;
pub mod dimension;
pub mod error;
pub mod group;
pub mod types;
pub mod variable;

pub use cf::{CfAttributes, FillValue};
pub use dimension::Dimension;
pub use error::Error;
pub use group::NetCDF4Group;
pub use rustyhdf5::AttrValue;
pub use types::NcType;
pub use variable::Variable;

use std::collections::HashMap;

/// A NetCDF-4 file reader.
///
/// Wraps a rustyhdf5 File and provides NetCDF-4 semantics: dimensions,
/// variables with CF attributes, groups, and type mapping.
pub struct NetCDF4File {
    hdf5: rustyhdf5::File,
}

impl NetCDF4File {
    /// Open a NetCDF-4 file from a filesystem path.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let hdf5 = rustyhdf5::File::open(path)?;
        Ok(Self { hdf5 })
    }

    /// Open a NetCDF-4 file from in-memory bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        let hdf5 = rustyhdf5::File::from_bytes(data)?;
        Ok(Self { hdf5 })
    }

    /// Get the _NCProperties root attribute, if present.
    ///
    /// This attribute is written by standard NetCDF-4 libraries and contains
    /// provenance information (library version, HDF5 version, etc.).
    pub fn nc_properties(&self) -> Result<Option<String>, Error> {
        let attrs = self.hdf5.root().attrs()?;
        match attrs.get("_NCProperties") {
            Some(AttrValue::String(s)) => Ok(Some(s.clone())),
            _ => Ok(None),
        }
    }

    /// List dimensions defined in the root group.
    pub fn dimensions(&self) -> Result<Vec<Dimension>, Error> {
        dimension::extract_dimensions_from_datasets(&self.hdf5.root(), &self.hdf5)
    }

    /// List all variables in the root group.
    pub fn variables(&self) -> Result<Vec<Variable<'_>>, Error> {
        let dims = self.dimensions()?;
        variable::build_variables(&self.hdf5.root(), &dims)
    }

    /// Get a specific variable by name from the root group.
    pub fn variable(&self, name: &str) -> Result<Variable<'_>, Error> {
        let dims = self.dimensions()?;
        let ds = self
            .hdf5
            .dataset(name)
            .map_err(|_| Error::VariableNotFound(name.to_string()))?;
        let shape = ds.shape()?;
        let var_dims = variable::match_dimensions_to_variable(&shape, &dims);
        Ok(Variable::new(name.to_string(), ds, var_dims))
    }

    /// Read all global (root group) attributes.
    pub fn global_attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        Ok(self.hdf5.root().attrs()?)
    }

    /// List subgroup names in the root group.
    pub fn group_names(&self) -> Result<Vec<String>, Error> {
        Ok(self.hdf5.root().groups()?)
    }

    /// Get a subgroup by name.
    pub fn group(&self, name: &str) -> Result<NetCDF4Group<'_>, Error> {
        let hdf5_group = self
            .hdf5
            .group(name)
            .map_err(|_| Error::GroupNotFound(name.to_string()))?;
        Ok(NetCDF4Group::new(
            name.to_string(),
            &self.hdf5,
            hdf5_group,
        ))
    }

    /// Access the underlying HDF5 file for advanced operations.
    pub fn hdf5_file(&self) -> &rustyhdf5::File {
        &self.hdf5
    }
}

impl std::fmt::Debug for NetCDF4File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetCDF4File")
            .field("hdf5", &self.hdf5)
            .finish()
    }
}
