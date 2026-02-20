//! Error types for the high-level API.

use std::fmt;

use purehdf5_format::error::FormatError;
use purehdf5_format::message_type::MessageType;

/// Errors that can occur when using the high-level API.
#[derive(Debug)]
pub enum Error {
    /// I/O error from the filesystem.
    Io(std::io::Error),
    /// Low-level format parsing error.
    Format(FormatError),
    /// The object at the given path is not a dataset.
    NotADataset(String),
    /// A required header message was not found.
    MissingMessage(MessageType),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "HDF5 format error: {e}"),
            Error::NotADataset(path) => write!(f, "not a dataset: {path}"),
            Error::MissingMessage(mt) => write!(f, "missing required message: {mt:?}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Error::Format(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
