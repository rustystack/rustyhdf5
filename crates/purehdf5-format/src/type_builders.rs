//! Builder types for HDF5 datatypes, attributes, datasets, and groups.
//!
//! Extracted from `file_writer.rs` to keep modules under the line limit.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::attribute::AttributeMessage;
use crate::chunked_write::ChunkOptions;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, StringPadding,
};

// ---- Datatype constructors ----

pub fn make_f64_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 64,
        exponent_location: 52,
        exponent_size: 11,
        mantissa_location: 0,
        mantissa_size: 52,
        exponent_bias: 1023,
    }
}

pub(crate) fn make_f32_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 32,
        exponent_location: 23,
        exponent_size: 8,
        mantissa_location: 0,
        mantissa_size: 23,
        exponent_bias: 127,
    }
}

pub(crate) fn make_i32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

pub(crate) fn make_i64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 64,
    }
}

pub(crate) fn make_u8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 8,
    }
}

// ---- Compound / Enum type builders ----

/// Builder for constructing HDF5 compound (struct) datatypes.
pub struct CompoundTypeBuilder {
    fields: Vec<(String, Datatype)>,
}

impl CompoundTypeBuilder {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a named field with the given datatype.
    pub fn field(mut self, name: &str, datatype: Datatype) -> Self {
        self.fields.push((name.to_string(), datatype));
        self
    }

    /// Add an f64 field.
    pub fn f64_field(self, name: &str) -> Self {
        self.field(name, make_f64_type())
    }
    /// Add an f32 field.
    pub fn f32_field(self, name: &str) -> Self {
        self.field(name, make_f32_type())
    }
    /// Add an i32 field.
    pub fn i32_field(self, name: &str) -> Self {
        self.field(name, make_i32_type())
    }
    /// Add an i64 field.
    pub fn i64_field(self, name: &str) -> Self {
        self.field(name, make_i64_type())
    }
    /// Add a u8 field.
    pub fn u8_field(self, name: &str) -> Self {
        self.field(name, make_u8_type())
    }

    /// Build the compound datatype.
    pub fn build(self) -> Datatype {
        let mut offset = 0u64;
        let mut members = Vec::with_capacity(self.fields.len());
        for (name, dt) in self.fields {
            let sz = dt.type_size();
            members.push(CompoundMember {
                name,
                byte_offset: offset,
                datatype: dt,
            });
            offset += sz as u64;
        }
        Datatype::Compound {
            size: offset as u32,
            members,
        }
    }
}

impl Default for CompoundTypeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing HDF5 enumeration datatypes.
pub struct EnumTypeBuilder {
    base_type: Datatype,
    members: Vec<EnumMember>,
}

impl EnumTypeBuilder {
    /// Create a new enum builder with i32 base type.
    pub fn i32_based() -> Self {
        Self {
            base_type: make_i32_type(),
            members: Vec::new(),
        }
    }

    /// Create a new enum builder with u8 base type.
    pub fn u8_based() -> Self {
        Self {
            base_type: make_u8_type(),
            members: Vec::new(),
        }
    }

    /// Add a named value.
    pub fn value(mut self, name: &str, val: i32) -> Self {
        self.members.push(EnumMember {
            name: name.to_string(),
            value: val.to_le_bytes().to_vec(),
        });
        self
    }

    /// Add a named u8 value.
    pub fn u8_value(mut self, name: &str, val: u8) -> Self {
        self.members.push(EnumMember {
            name: name.to_string(),
            value: vec![val],
        });
        self
    }

    /// Build the enumeration datatype.
    pub fn build(self) -> Datatype {
        let size = self.base_type.type_size();
        Datatype::Enumeration {
            size,
            base_type: Box::new(self.base_type),
            members: self.members,
        }
    }
}

// ---- Attribute helper ----

pub(crate) fn build_attr_message(name: &str, value: &AttrValue) -> AttributeMessage {
    match value {
        AttrValue::F64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_f64_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::F64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: make_f64_type(),
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::I64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_i64_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::I64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: make_i64_type(),
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::U64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: Datatype::FixedPoint {
                size: 8,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: false,
                bit_offset: 0,
                bit_precision: 64,
            },
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::String(s) => {
            let bytes = s.as_bytes();
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: bytes.len() as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: scalar_ds(),
                raw_data: bytes.to_vec(),
            }
        }
        AttrValue::StringArray(arr) => {
            let max_len = arr.iter().map(|s| s.len()).max().unwrap_or(0);
            let mut raw = Vec::new();
            for s in arr {
                let mut b = s.as_bytes().to_vec();
                b.resize(max_len, 0);
                raw.extend_from_slice(&b);
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: max_len as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
    }
}

pub(crate) fn scalar_ds() -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Scalar,
        rank: 0,
        dimensions: vec![],
        max_dimensions: None,
    }
}

pub(crate) fn simple_1d(n: u64) -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![n],
        max_dimensions: None,
    }
}

// ---- Attribute values ----

/// Convenient attribute values for the write API.
#[derive(Debug, Clone)]
pub enum AttrValue {
    F64(f64),
    F64Array(Vec<f64>),
    I64(i64),
    I64Array(Vec<i64>),
    U64(u64),
    String(String),
    StringArray(Vec<String>),
}

// ---- Dataset builder ----

/// Configuration for SHINES provenance metadata.
#[cfg(feature = "provenance")]
#[derive(Debug, Clone)]
pub struct ProvenanceConfig {
    pub creator: String,
    pub timestamp: String,
    pub source: Option<String>,
}

/// Builder for datasets.
pub struct DatasetBuilder {
    pub(crate) name: String,
    pub(crate) datatype: Option<Datatype>,
    pub(crate) shape: Option<Vec<u64>>,
    pub(crate) maxshape: Option<Vec<u64>>,
    pub(crate) data: Option<Vec<u8>>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
    pub(crate) chunk_options: ChunkOptions,
    #[cfg(feature = "provenance")]
    pub(crate) provenance: Option<ProvenanceConfig>,
}

impl DatasetBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datatype: None,
            shape: None,
            maxshape: None,
            data: None,
            attrs: Vec::new(),
            chunk_options: ChunkOptions::default(),
            #[cfg(feature = "provenance")]
            provenance: None,
        }
    }

    pub fn with_f64_data(&mut self, data: &[f64]) -> &mut Self {
        self.datatype = Some(make_f64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_f32_data(&mut self, data: &[f32]) -> &mut Self {
        self.datatype = Some(make_f32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i32_data(&mut self, data: &[i32]) -> &mut Self {
        self.datatype = Some(make_i32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i64_data(&mut self, data: &[i64]) -> &mut Self {
        self.datatype = Some(make_i64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u8_data(&mut self, data: &[u8]) -> &mut Self {
        self.datatype = Some(make_u8_type());
        self.data = Some(data.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    /// Write a compound (struct) dataset.
    pub fn with_compound_data(
        &mut self,
        datatype: Datatype,
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    /// Write an enum dataset with i32 values.
    pub fn with_enum_i32_data(&mut self, datatype: Datatype, values: &[i32]) -> &mut Self {
        self.datatype = Some(datatype);
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(raw);
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write an enum dataset with u8 values.
    pub fn with_enum_u8_data(&mut self, datatype: Datatype, values: &[u8]) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(values.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write an array-typed dataset.
    pub fn with_array_data(
        &mut self,
        base_type: Datatype,
        array_dims: &[u32],
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(Datatype::Array {
            base_type: Box::new(base_type),
            dimensions: array_dims.to_vec(),
        });
        self.data = Some(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    pub fn with_shape(&mut self, shape: &[u64]) -> &mut Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Set maximum dimensions for a resizable dataset.
    /// Use `u64::MAX` for unlimited dimensions.
    pub fn with_maxshape(&mut self, maxshape: &[u64]) -> &mut Self {
        self.maxshape = Some(maxshape.to_vec());
        self
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.attrs.push((name.to_string(), value));
        self
    }

    /// Enable chunked storage with given chunk dimensions.
    pub fn with_chunks(&mut self, chunk_dims: &[u64]) -> &mut Self {
        self.chunk_options.chunk_dims = Some(chunk_dims.to_vec());
        self
    }

    /// Enable deflate compression (implies chunked if not already set).
    pub fn with_deflate(&mut self, level: u32) -> &mut Self {
        self.chunk_options.deflate_level = Some(level);
        self
    }

    /// Enable shuffle filter (usually combined with deflate).
    pub fn with_shuffle(&mut self) -> &mut Self {
        self.chunk_options.shuffle = true;
        self
    }

    /// Enable fletcher32 checksum.
    pub fn with_fletcher32(&mut self) -> &mut Self {
        self.chunk_options.fletcher32 = true;
        self
    }

    /// Attach SHINES provenance metadata (SHA-256, creator, timestamp).
    ///
    /// The SHA-256 hash of the raw dataset bytes is computed automatically
    /// during file serialization and stored as `_provenance_sha256`.
    #[cfg(feature = "provenance")]
    pub fn with_provenance(
        &mut self,
        creator: &str,
        timestamp: &str,
        source: Option<&str>,
    ) -> &mut Self {
        self.provenance = Some(ProvenanceConfig {
            creator: creator.to_string(),
            timestamp: timestamp.to_string(),
            source: source.map(|s| s.to_string()),
        });
        self
    }
}

// ---- Group builder ----

/// Builder for groups.
pub struct GroupBuilder {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
}

impl GroupBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datasets: Vec::new(),
            attrs: Vec::new(),
        }
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.datasets.push(DatasetBuilder::new(name));
        self.datasets.last_mut().unwrap()
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.attrs.push((name.to_string(), value));
    }

    /// Consume the builder, returning a FinishedGroup to add to FileWriter.
    pub fn finish(self) -> FinishedGroup {
        FinishedGroup {
            name: self.name,
            datasets: self.datasets,
            attrs: self.attrs,
        }
    }
}

/// A finished group ready for the file writer.
pub struct FinishedGroup {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
}
