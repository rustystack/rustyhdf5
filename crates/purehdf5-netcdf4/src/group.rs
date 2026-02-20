//! NetCDF-4 group representation.
//!
//! NetCDF-4 uses HDF5 groups as NetCDF groups. Each group can contain
//! dimensions, variables, attributes, and subgroups.

use std::collections::HashMap;

use purehdf5::AttrValue;

use crate::dimension::{self, Dimension};
use crate::error::Error;
use crate::variable::{self, Variable};

/// A NetCDF-4 group corresponding to an HDF5 group.
pub struct NetCDF4Group<'f> {
    /// Group name.
    name: String,
    /// Underlying HDF5 file.
    file: &'f purehdf5::File,
    /// Underlying HDF5 group.
    hdf5_group: purehdf5::Group<'f>,
}

impl<'f> NetCDF4Group<'f> {
    /// Create a new NetCDF4Group from an HDF5 group.
    pub(crate) fn new(
        name: String,
        file: &'f purehdf5::File,
        hdf5_group: purehdf5::Group<'f>,
    ) -> Self {
        Self {
            name,
            file,
            hdf5_group,
        }
    }

    /// Group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// List dimensions defined in this group.
    pub fn dimensions(&self) -> Result<Vec<Dimension>, Error> {
        dimension::extract_dimensions_from_datasets(&self.hdf5_group, self.file)
    }

    /// List variables in this group.
    pub fn variables(&self) -> Result<Vec<Variable<'f>>, Error> {
        let dims = self.dimensions()?;
        variable::build_variables(&self.hdf5_group, &dims)
    }

    /// Get a specific variable by name.
    pub fn variable(&self, name: &str) -> Result<Variable<'f>, Error> {
        let dims = self.dimensions()?;
        let ds = self
            .hdf5_group
            .dataset(name)
            .map_err(|_| Error::VariableNotFound(name.to_string()))?;
        let shape = ds.shape()?;
        let var_dims = crate::variable::match_dimensions_to_variable(&shape, &dims);
        Ok(Variable::new(name.to_string(), ds, var_dims))
    }

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        Ok(self.hdf5_group.attrs()?)
    }

    /// List subgroup names.
    pub fn group_names(&self) -> Result<Vec<String>, Error> {
        Ok(self.hdf5_group.groups()?)
    }

    /// Get a subgroup by name.
    pub fn group(&self, name: &str) -> Result<NetCDF4Group<'f>, Error> {
        let hdf5_group = self
            .hdf5_group
            .group(name)
            .map_err(|_| Error::GroupNotFound(name.to_string()))?;
        Ok(NetCDF4Group::new(name.to_string(), self.file, hdf5_group))
    }

    /// List dataset (variable) names in this group.
    pub fn variable_names(&self) -> Result<Vec<String>, Error> {
        Ok(self.hdf5_group.datasets()?)
    }
}

impl std::fmt::Debug for NetCDF4Group<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetCDF4Group")
            .field("name", &self.name)
            .finish()
    }
}
