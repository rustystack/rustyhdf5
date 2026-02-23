//! NetCDF-4 data type mapping.
//!
//! Maps HDF5 types to NetCDF type names and provides type classification.

use rustyhdf5::DType;

/// NetCDF-4 data types corresponding to the standard NetCDF type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NcType {
    /// NC_BYTE: signed 8-bit integer
    Byte,
    /// NC_UBYTE: unsigned 8-bit integer
    UByte,
    /// NC_SHORT: signed 16-bit integer
    Short,
    /// NC_USHORT: unsigned 16-bit integer
    UShort,
    /// NC_INT: signed 32-bit integer
    Int,
    /// NC_UINT: unsigned 32-bit integer
    UInt,
    /// NC_INT64: signed 64-bit integer
    Int64,
    /// NC_UINT64: unsigned 64-bit integer
    UInt64,
    /// NC_FLOAT: 32-bit floating point
    Float,
    /// NC_DOUBLE: 64-bit floating point
    Double,
    /// NC_STRING: variable-length string
    String,
    /// NC_CHAR: fixed-length string / character data
    Char,
}

impl std::fmt::Display for NcType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NcType::Byte => write!(f, "NC_BYTE"),
            NcType::UByte => write!(f, "NC_UBYTE"),
            NcType::Short => write!(f, "NC_SHORT"),
            NcType::UShort => write!(f, "NC_USHORT"),
            NcType::Int => write!(f, "NC_INT"),
            NcType::UInt => write!(f, "NC_UINT"),
            NcType::Int64 => write!(f, "NC_INT64"),
            NcType::UInt64 => write!(f, "NC_UINT64"),
            NcType::Float => write!(f, "NC_FLOAT"),
            NcType::Double => write!(f, "NC_DOUBLE"),
            NcType::String => write!(f, "NC_STRING"),
            NcType::Char => write!(f, "NC_CHAR"),
        }
    }
}

/// Map a rustyhdf5 DType to a NetCDF type.
pub fn dtype_to_nctype(dtype: &DType) -> NcType {
    match dtype {
        DType::I8 => NcType::Byte,
        DType::U8 => NcType::UByte,
        DType::I16 => NcType::Short,
        DType::U16 => NcType::UShort,
        DType::I32 => NcType::Int,
        DType::U32 => NcType::UInt,
        DType::I64 => NcType::Int64,
        DType::U64 => NcType::UInt64,
        DType::F32 => NcType::Float,
        DType::F64 => NcType::Double,
        DType::String | DType::VariableLengthString => NcType::String,
        _ => NcType::Char, // fallback for other types
    }
}
