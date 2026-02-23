//! Error types for the NetCDF-4 reader.

use std::fmt;

/// Errors that can occur when reading NetCDF-4 files.
#[derive(Debug)]
pub enum Error {
    /// I/O error from the filesystem.
    Io(std::io::Error),
    /// Low-level HDF5 format error.
    Hdf5(rustyhdf5::Error),
    /// The file is not a valid NetCDF-4 file (missing _NCProperties or conventions).
    NotNetCDF4(String),
    /// A required dimension was not found.
    DimensionNotFound(String),
    /// A required variable was not found.
    VariableNotFound(String),
    /// A required group was not found.
    GroupNotFound(String),
    /// Data type conversion error.
    TypeError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Hdf5(e) => write!(f, "HDF5 error: {e}"),
            Error::NotNetCDF4(msg) => write!(f, "not a NetCDF-4 file: {msg}"),
            Error::DimensionNotFound(name) => write!(f, "dimension not found: {name}"),
            Error::VariableNotFound(name) => write!(f, "variable not found: {name}"),
            Error::GroupNotFound(name) => write!(f, "group not found: {name}"),
            Error::TypeError(msg) => write!(f, "type error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Hdf5(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<rustyhdf5::Error> for Error {
    fn from(e: rustyhdf5::Error) -> Self {
        Error::Hdf5(e)
    }
}
