//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, and inline attributes.

use crate::attribute::AttributeMessage;
use crate::chunked_write::{ChunkOptions, build_chunked_data_at_ext};
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, StringPadding};
use crate::error::FormatError;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::superblock::Superblock;

const OFFSET_SIZE: u8 = 8;
const LENGTH_SIZE: u8 = 8;
const SUPERBLOCK_SIZE: usize = 48;

// ---- Datatype constructors ----

fn make_f64_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 8, byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0, bit_precision: 64,
        exponent_location: 52, exponent_size: 11,
        mantissa_location: 0, mantissa_size: 52, exponent_bias: 1023,
    }
}

fn make_f32_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 4, byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0, bit_precision: 32,
        exponent_location: 23, exponent_size: 8,
        mantissa_location: 0, mantissa_size: 23, exponent_bias: 127,
    }
}

fn make_i32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4, byte_order: DatatypeByteOrder::LittleEndian,
        signed: true, bit_offset: 0, bit_precision: 32,
    }
}

fn make_i64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8, byte_order: DatatypeByteOrder::LittleEndian,
        signed: true, bit_offset: 0, bit_precision: 64,
    }
}

fn make_u8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1, byte_order: DatatypeByteOrder::LittleEndian,
        signed: false, bit_offset: 0, bit_precision: 8,
    }
}

// ---- Compound / Enum / Array type builders ----

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
    pub fn f64_field(self, name: &str) -> Self { self.field(name, make_f64_type()) }
    /// Add an f32 field.
    pub fn f32_field(self, name: &str) -> Self { self.field(name, make_f32_type()) }
    /// Add an i32 field.
    pub fn i32_field(self, name: &str) -> Self { self.field(name, make_i32_type()) }
    /// Add an i64 field.
    pub fn i64_field(self, name: &str) -> Self { self.field(name, make_i64_type()) }
    /// Add a u8 field.
    pub fn u8_field(self, name: &str) -> Self { self.field(name, make_u8_type()) }

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
    fn default() -> Self { Self::new() }
}

/// Builder for constructing HDF5 enumeration datatypes.
pub struct EnumTypeBuilder {
    base_type: Datatype,
    members: Vec<EnumMember>,
}

impl EnumTypeBuilder {
    /// Create a new enum builder with i32 base type.
    pub fn i32_based() -> Self {
        Self { base_type: make_i32_type(), members: Vec::new() }
    }

    /// Create a new enum builder with u8 base type.
    pub fn u8_based() -> Self {
        Self { base_type: make_u8_type(), members: Vec::new() }
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

fn build_attr_message(name: &str, value: &AttrValue) -> AttributeMessage {
    match value {
        AttrValue::F64(v) => AttributeMessage {
            name: name.to_string(), datatype: make_f64_type(),
            dataspace: scalar_ds(), raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::F64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr { raw.extend_from_slice(&v.to_le_bytes()); }
            AttributeMessage {
                name: name.to_string(), datatype: make_f64_type(),
                dataspace: simple_1d(arr.len() as u64), raw_data: raw,
            }
        }
        AttrValue::I64(v) => AttributeMessage {
            name: name.to_string(), datatype: make_i64_type(),
            dataspace: scalar_ds(), raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::I64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr { raw.extend_from_slice(&v.to_le_bytes()); }
            AttributeMessage {
                name: name.to_string(), datatype: make_i64_type(),
                dataspace: simple_1d(arr.len() as u64), raw_data: raw,
            }
        }
        AttrValue::U64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: Datatype::FixedPoint {
                size: 8, byte_order: DatatypeByteOrder::LittleEndian,
                signed: false, bit_offset: 0, bit_precision: 64,
            },
            dataspace: scalar_ds(), raw_data: v.to_le_bytes().to_vec(),
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
                dataspace: scalar_ds(), raw_data: bytes.to_vec(),
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
                    size: max_len as u32, padding: StringPadding::NullPad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: simple_1d(arr.len() as u64), raw_data: raw,
            }
        }
    }
}

fn scalar_ds() -> Dataspace {
    Dataspace { space_type: DataspaceType::Scalar, rank: 0, dimensions: vec![], max_dimensions: None }
}

fn simple_1d(n: u64) -> Dataspace {
    Dataspace { space_type: DataspaceType::Simple, rank: 1, dimensions: vec![n], max_dimensions: None }
}

// ---- OH builders ----

fn build_chunked_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, 0x0a], 0x01);
    w.add_message(MessageType::DataLayout, layout_message.to_vec());
    if let Some(pm) = pipeline_message {
        w.add_message(MessageType::FilterPipeline, pm.to_vec());
    }
    for attr in attrs {
        w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
    }
    w.serialize()
}

fn build_dataset_oh(
    dt: &Datatype, ds: &Dataspace, data_addr: u64, data_size: u64,
    attrs: &[AttributeMessage],
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    // Datatype with "constant" flag (0x01)
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    // FillValue v3: version(1)=3, flags(1)=0x0a (alloc_time=2/late, write_time=2/if_defined, defined=0)
    w.add_message_with_flags(MessageType::FillValue, vec![3, 0x0a], 0x01);
    // Data layout: v4 contiguous (matching h5py reference)
    let mut dl = Vec::new();
    dl.push(4); // version
    dl.push(1); // class = contiguous
    dl.extend_from_slice(&data_addr.to_le_bytes());
    dl.extend_from_slice(&data_size.to_le_bytes());
    w.add_message(MessageType::DataLayout, dl);
    for attr in attrs {
        w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
    }
    w.serialize()
}

fn build_group_oh(links: &[LinkMessage], attrs: &[AttributeMessage]) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    // LinkInfo message: version=0, flags=0, fh_addr=UNDEF, btree_addr=UNDEF
    // This tells HDF5 C library this is a group (compact storage).
    let mut li = Vec::new();
    li.push(0); // version
    li.push(0); // flags (no creation order tracking)
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF (compact)
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
    w.add_message(MessageType::LinkInfo, li);
    for link in links {
        w.add_message(MessageType::Link, link.serialize(OFFSET_SIZE));
    }
    for attr in attrs {
        w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
    }
    w.serialize()
}

fn make_link(name: &str, addr: u64) -> LinkMessage {
    LinkMessage {
        name: name.to_string(),
        link_target: LinkTarget::Hard { object_header_address: addr },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
}

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

/// Builder for datasets.
pub struct DatasetBuilder {
    name: String,
    datatype: Option<Datatype>,
    shape: Option<Vec<u64>>,
    maxshape: Option<Vec<u64>>,
    data: Option<Vec<u8>>,
    attrs: Vec<(String, AttrValue)>,
    chunk_options: ChunkOptions,
}

impl DatasetBuilder {
    fn new(name: &str) -> Self {
        Self { name: name.to_string(), datatype: None, shape: None, maxshape: None, data: None, attrs: Vec::new(), chunk_options: ChunkOptions::default() }
    }

    pub fn with_f64_data(&mut self, data: &[f64]) -> &mut Self {
        self.datatype = Some(make_f64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data { b.extend_from_slice(&v.to_le_bytes()); }
        self.data = Some(b);
        if self.shape.is_none() { self.shape = Some(vec![data.len() as u64]); }
        self
    }

    pub fn with_f32_data(&mut self, data: &[f32]) -> &mut Self {
        self.datatype = Some(make_f32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data { b.extend_from_slice(&v.to_le_bytes()); }
        self.data = Some(b);
        if self.shape.is_none() { self.shape = Some(vec![data.len() as u64]); }
        self
    }

    pub fn with_i32_data(&mut self, data: &[i32]) -> &mut Self {
        self.datatype = Some(make_i32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data { b.extend_from_slice(&v.to_le_bytes()); }
        self.data = Some(b);
        if self.shape.is_none() { self.shape = Some(vec![data.len() as u64]); }
        self
    }

    pub fn with_i64_data(&mut self, data: &[i64]) -> &mut Self {
        self.datatype = Some(make_i64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data { b.extend_from_slice(&v.to_le_bytes()); }
        self.data = Some(b);
        if self.shape.is_none() { self.shape = Some(vec![data.len() as u64]); }
        self
    }

    pub fn with_u8_data(&mut self, data: &[u8]) -> &mut Self {
        self.datatype = Some(make_u8_type());
        self.data = Some(data.to_vec());
        if self.shape.is_none() { self.shape = Some(vec![data.len() as u64]); }
        self
    }

    /// Write a compound (struct) dataset.
    ///
    /// `datatype` should be built with `CompoundTypeBuilder`.
    /// `raw_data` is the packed binary data for all elements.
    /// `num_elements` is the number of compound elements.
    pub fn with_compound_data(&mut self, datatype: Datatype, raw_data: Vec<u8>, num_elements: u64) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(raw_data);
        if self.shape.is_none() { self.shape = Some(vec![num_elements]); }
        self
    }

    /// Write an enum dataset.
    ///
    /// `datatype` should be built with `EnumTypeBuilder`.
    /// `values` are the i32 enum values to write.
    pub fn with_enum_i32_data(&mut self, datatype: Datatype, values: &[i32]) -> &mut Self {
        self.datatype = Some(datatype);
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values { raw.extend_from_slice(&v.to_le_bytes()); }
        self.data = Some(raw);
        if self.shape.is_none() { self.shape = Some(vec![values.len() as u64]); }
        self
    }

    /// Write an enum dataset with u8 values.
    pub fn with_enum_u8_data(&mut self, datatype: Datatype, values: &[u8]) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(values.to_vec());
        if self.shape.is_none() { self.shape = Some(vec![values.len() as u64]); }
        self
    }

    /// Write an array-typed dataset.
    ///
    /// `base_type` is the element type (e.g., f64), `array_dims` is the fixed array shape
    /// per element, `raw_data` is the packed data, `num_elements` is the dataset element count.
    pub fn with_array_data(&mut self, base_type: Datatype, array_dims: &[u32], raw_data: Vec<u8>, num_elements: u64) -> &mut Self {
        self.datatype = Some(Datatype::Array {
            base_type: Box::new(base_type),
            dimensions: array_dims.to_vec(),
        });
        self.data = Some(raw_data);
        if self.shape.is_none() { self.shape = Some(vec![num_elements]); }
        self
    }

    pub fn with_shape(&mut self, shape: &[u64]) -> &mut Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Set maximum dimensions for a resizable dataset.
    /// Use `u64::MAX` for unlimited dimensions.
    /// Implies chunked storage if not already set.
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
}

/// Builder for groups.
pub struct GroupBuilder {
    name: String,
    datasets: Vec<DatasetBuilder>,
    attrs: Vec<(String, AttrValue)>,
}

impl GroupBuilder {
    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.datasets.push(DatasetBuilder::new(name));
        self.datasets.last_mut().unwrap()
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.attrs.push((name.to_string(), value));
    }

    /// Consume the builder, returning a FinishedGroup to add to FileWriter.
    pub fn finish(self) -> FinishedGroup {
        FinishedGroup { name: self.name, datasets: self.datasets, attrs: self.attrs }
    }
}

/// A finished group ready for the file writer.
pub struct FinishedGroup {
    name: String,
    datasets: Vec<DatasetBuilder>,
    attrs: Vec<(String, AttrValue)>,
}

/// The main file creation API.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: Vec<(String, AttrValue)>,
    groups: Vec<FinishedGroup>,
}

impl Default for FileWriter {
    fn default() -> Self { Self::new() }
}

impl FileWriter {
    pub fn new() -> Self {
        Self { root_datasets: Vec::new(), root_attrs: Vec::new(), groups: Vec::new() }
    }

    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder { name: name.to_string(), datasets: Vec::new(), attrs: Vec::new() }
    }

    pub fn add_group(&mut self, group: FinishedGroup) {
        self.groups.push(group);
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.root_datasets.push(DatasetBuilder::new(name));
        self.root_datasets.last_mut().unwrap()
    }

    pub fn set_root_attr(&mut self, name: &str, value: AttrValue) {
        self.root_attrs.push((name.to_string(), value));
    }

    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        // Extract structured info from builders
        struct DsFlat {
            name: String,
            dt: Datatype,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<u64>>,
        }
        struct GrpFlat {
            name: String,
            attrs: Vec<AttributeMessage>,
            ds_indices: Vec<usize>,
        }

        let mut all_ds: Vec<DsFlat> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();
        let mut root_ds_indices: Vec<usize> = Vec::new();

        for db in self.root_datasets {
            let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
            let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
            let raw = db.data.ok_or(FormatError::DatasetMissingData)?;
            let max_dimensions = db.maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() { DataspaceType::Scalar } else { DataspaceType::Simple },
                rank: shape.len() as u8, dimensions: shape, max_dimensions,
            };
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs { attrs.push(build_attr_message(n, v)); }
            let idx = all_ds.len();
            root_ds_indices.push(idx);
            all_ds.push(DsFlat { name: db.name, dt, ds: dspace, raw, attrs, chunk_options: db.chunk_options, maxshape: db.maxshape });
        }

        for g in self.groups.into_iter() {
            let mut gattrs = Vec::new();
            for (n, v) in &g.attrs { gattrs.push(build_attr_message(n, v)); }
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
                let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
                let raw = db.data.ok_or(FormatError::DatasetMissingData)?;
                let max_dimensions = db.maxshape.clone();
                let dspace = Dataspace {
                    space_type: if shape.is_empty() { DataspaceType::Scalar } else { DataspaceType::Simple },
                    rank: shape.len() as u8, dimensions: shape, max_dimensions,
                };
                let mut attrs = Vec::new();
                for (n, v) in &db.attrs { attrs.push(build_attr_message(n, v)); }
                let idx = all_ds.len();
                ds_idx.push(idx);
                all_ds.push(DsFlat { name: db.name, dt, ds: dspace, raw, attrs, chunk_options: db.chunk_options, maxshape: db.maxshape });
            }
            groups.push(GrpFlat { name: g.name, attrs: gattrs, ds_indices: ds_idx });
        }

        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        for (n, v) in &self.root_attrs { root_attrs.push(build_attr_message(n, v)); }

        // Separate contiguous vs chunked datasets
        let is_chunked: Vec<bool> = all_ds.iter().map(|d| {
            d.chunk_options.is_chunked() || d.maxshape.is_some()
        }).collect();

        // For contiguous datasets: OH size is stable (address doesn't affect size)
        // For chunked datasets: we need to build the chunked data blob first to know
        // the layout message content, then build the OH.
        //
        // Strategy: Two passes.
        // Pass 1: Compute group OH sizes (stable), contiguous dataset OH sizes (stable).
        //         For chunked datasets, build a dummy OH to get approximate size.
        // Pass 2: Assign addresses, build chunked blobs at correct addresses, build final OHs.

        let group_oh_sizes: Vec<usize> = groups.iter().map(|g| {
            let dummy_links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| {
                make_link(&all_ds[i].name, 0)
            }).collect();
            build_group_oh(&dummy_links, &g.attrs).len()
        }).collect();

        let root_dummy_links: Vec<LinkMessage> = {
            let mut links = Vec::new();
            for &i in &root_ds_indices { links.push(make_link(&all_ds[i].name, 0)); }
            for g in &groups { links.push(make_link(&g.name, 0)); }
            links
        };
        let root_oh_size = build_group_oh(&root_dummy_links, &root_attrs).len();

        // Two-pass address assignment.
        // Pass 1: Build chunked data blobs with dummy base address to learn OH sizes.
        // Pass 2: Rebuild with correct addresses.

        struct DataBlob {
            data: Vec<u8>,
            oh_bytes: Vec<u8>,
        }

        // Pass 1: build blobs with dummy addresses to get actual OH sizes
        let mut dummy_blobs: Vec<DataBlob> = Vec::new();
        let mut dummy_cursor = 0u64; // dummy, just for sizing
        for (i, d) in all_ds.iter().enumerate() {
            if is_chunked[i] {
                let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                let elem_size = d.dt.type_size() as usize;
                let result = build_chunked_data_at_ext(
                    &d.raw, &d.ds.dimensions, &chunk_dims, elem_size,
                    &d.chunk_options, dummy_cursor, d.maxshape.as_deref(),
                )?;
                dummy_cursor += result.data_bytes.len() as u64;
                let oh = build_chunked_dataset_oh(
                    &d.dt, &d.ds, &result.layout_message,
                    result.pipeline_message.as_deref(), &d.attrs,
                );
                dummy_blobs.push(DataBlob { data: result.data_bytes, oh_bytes: oh });
            } else {
                let oh = build_dataset_oh(
                    &d.dt, &d.ds, 0, d.raw.len() as u64, &d.attrs,
                );
                dummy_blobs.push(DataBlob { data: d.raw.clone(), oh_bytes: oh });
            }
        }

        let actual_ds_oh_sizes: Vec<usize> = dummy_blobs.iter().map(|b| b.oh_bytes.len()).collect();

        // Pass 2: compute real addresses
        let root_group_addr = SUPERBLOCK_SIZE as u64;
        let mut cursor2 = SUPERBLOCK_SIZE + root_oh_size;

        let group_addrs2: Vec<u64> = group_oh_sizes.iter().map(|&sz| {
            let addr = cursor2 as u64;
            cursor2 += sz;
            addr
        }).collect();

        let ds_oh_addrs2: Vec<u64> = actual_ds_oh_sizes.iter().map(|&sz| {
            let addr = cursor2 as u64;
            cursor2 += sz;
            addr
        }).collect();

        // Now rebuild data blobs with correct base addresses
        let mut ds_blobs2: Vec<DataBlob> = Vec::new();
        for (i, d) in all_ds.iter().enumerate() {
            if is_chunked[i] {
                let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                let elem_size = d.dt.type_size() as usize;
                let base_address = cursor2 as u64;
                let result = build_chunked_data_at_ext(
                    &d.raw, &d.ds.dimensions, &chunk_dims, elem_size,
                    &d.chunk_options, base_address, d.maxshape.as_deref(),
                )?;
                cursor2 += result.data_bytes.len();

                let oh = build_chunked_dataset_oh(
                    &d.dt, &d.ds, &result.layout_message,
                    result.pipeline_message.as_deref(), &d.attrs,
                );

                ds_blobs2.push(DataBlob { data: result.data_bytes, oh_bytes: oh });
            } else {
                let oh = build_dataset_oh(
                    &d.dt, &d.ds, cursor2 as u64, d.raw.len() as u64, &d.attrs,
                );
                let data = d.raw.clone();
                cursor2 += data.len();
                ds_blobs2.push(DataBlob { data, oh_bytes: oh });
            }
        }

        // Verify OH sizes are stable between pass 1 and pass 2
        let actual_ds_oh_sizes2: Vec<usize> = ds_blobs2.iter().map(|b| b.oh_bytes.len()).collect();
        debug_assert_eq!(actual_ds_oh_sizes, actual_ds_oh_sizes2);

        let ds_oh_addrs_final = &ds_oh_addrs2;
        let group_addrs_final = &group_addrs2;
        let eof_addr2 = cursor2 as u64;

        let mut buf = Vec::with_capacity(cursor2);

        let sb = Superblock {
            version: 3,
            offset_size: OFFSET_SIZE,
            length_size: LENGTH_SIZE,
            base_address: 0,
            eof_address: eof_addr2,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(u64::MAX),
            checksum: None,
        };
        buf.extend_from_slice(&sb.serialize());

        // Root group OH
        let mut root_links: Vec<LinkMessage> = Vec::new();
        for &i in &root_ds_indices {
            root_links.push(make_link(&all_ds[i].name, ds_oh_addrs_final[i]));
        }
        for (gi, g) in groups.iter().enumerate() {
            root_links.push(make_link(&g.name, group_addrs_final[gi]));
        }
        buf.extend_from_slice(&build_group_oh(&root_links, &root_attrs));

        // Group OHs
        for g in &groups {
            let links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| {
                make_link(&all_ds[i].name, ds_oh_addrs_final[i])
            }).collect();
            buf.extend_from_slice(&build_group_oh(&links, &g.attrs));
        }

        // Dataset OHs
        for blob in &ds_blobs2 {
            buf.extend_from_slice(&blob.oh_bytes);
        }

        // Data
        for blob in &ds_blobs2 {
            buf.extend_from_slice(&blob.data);
        }

        debug_assert_eq!(buf.len(), cursor2);
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_layout::DataLayout;
    use crate::data_read;
    use crate::group_v2::resolve_path_any;
    use crate::object_header::ObjectHeader;
    use crate::signature;

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let oh = ObjectHeader::parse(bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
        let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
        let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        data_read::read_as_f64(&raw, &dt).unwrap()
    }

    fn read_dataset_i32(bytes: &[u8], path: &str) -> Vec<i32> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
        let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
        let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        data_read::read_as_i32(&raw, &dt).unwrap()
    }

    // ---- Serialization round-trip tests ----

    #[test]
    fn serialize_parse_superblock_v3() {
        let sb = Superblock {
            version: 3, offset_size: 8, length_size: 8,
            base_address: 0, eof_address: 1024, root_group_address: 48,
            group_leaf_node_k: None, group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None, driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(u64::MAX),
            checksum: None,
        };
        let bytes = sb.serialize();
        let parsed = Superblock::parse(&bytes, 0).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.offset_size, 8);
        assert_eq!(parsed.eof_address, 1024);
        assert_eq!(parsed.root_group_address, 48);
    }

    #[test]
    fn serialize_parse_datatype_f64() {
        let dt = make_f64_type();
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_datatype_i32() {
        let dt = make_i32_type();
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_dataspace_1d() {
        let ds = simple_1d(5);
        let bytes = ds.serialize(8);
        let parsed = Dataspace::parse(&bytes, 8).unwrap();
        assert_eq!(parsed.space_type, DataspaceType::Simple);
        assert_eq!(parsed.dimensions, vec![5]);
    }

    #[test]
    fn serialize_parse_dataspace_scalar() {
        let ds = scalar_ds();
        let bytes = ds.serialize(8);
        let parsed = Dataspace::parse(&bytes, 8).unwrap();
        assert_eq!(parsed.space_type, DataspaceType::Scalar);
        assert_eq!(parsed.rank, 0);
    }

    #[test]
    fn serialize_parse_link_message() {
        let lm = make_link("test_link", 0x1234);
        let bytes = lm.serialize(8);
        let parsed = LinkMessage::parse(&bytes, 8).unwrap();
        assert_eq!(parsed.name, "test_link");
        assert_eq!(parsed.link_target, LinkTarget::Hard { object_header_address: 0x1234 });
    }

    #[test]
    fn serialize_parse_attribute_f64() {
        let attr = build_attr_message("temp", &AttrValue::F64(98.6));
        let bytes = attr.serialize(8);
        let parsed = AttributeMessage::parse(&bytes, 8).unwrap();
        assert_eq!(parsed.name, "temp");
        let vals = parsed.read_as_f64().unwrap();
        assert!((vals[0] - 98.6).abs() < 1e-10);
    }

    // ---- FileWriter tests ----

    #[test]
    fn empty_file_root_group_only() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        assert_eq!(sb.version, 3);
        assert_eq!(oh.version, 2);
    }

    #[test]
    fn file_with_f64_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();
        let vals = read_dataset_f64(&bytes, "data");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn file_with_i32_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("ints").with_i32_data(&[10, 20, 30]);
        let bytes = fw.finish().unwrap();
        let vals = read_dataset_i32(&bytes, "ints");
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn file_with_dataset_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        // Read dataset
        let vals = read_dataset_f64(&bytes, "data");
        assert_eq!(vals, vec![1.0, 2.0]);
        // Read attr
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "scale");
        let v = attrs[0].read_as_f64().unwrap();
        assert!((v[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn file_with_group_and_dataset() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
        let g = gb.finish();
        fw.add_group(g);
        let bytes = fw.finish().unwrap();
        let vals = read_dataset_f64(&bytes, "grp/vals");
        assert_eq!(vals, vec![10.0, 20.0]);
    }

    #[test]
    fn file_with_root_attr() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        let attrs = crate::attribute::extract_attributes(&oh, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "version");
        let v = attrs[0].read_as_i64().unwrap();
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn file_with_multiple_datasets() {
        let mut fw = FileWriter::new();
        fw.create_dataset("a").with_f64_data(&[1.0]);
        fw.create_dataset("b").with_f64_data(&[2.0]);
        fw.create_dataset("c").with_f64_data(&[3.0]);
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "a"), vec![1.0]);
        assert_eq!(read_dataset_f64(&bytes, "b"), vec![2.0]);
        assert_eq!(read_dataset_f64(&bytes, "c"), vec![3.0]);
    }

    #[test]
    fn file_with_2d_data() {
        let mut fw = FileWriter::new();
        fw.create_dataset("matrix")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[2, 3]);
        let bytes = fw.finish().unwrap();
        let vals = read_dataset_f64(&bytes, "matrix");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ---- Compound / Enum / Array write tests ----

    fn read_dataset_raw(bytes: &[u8], path: &str) -> (Vec<u8>, Datatype) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
        let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
        let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        (raw, dt)
    }

    #[test]
    fn file_with_compound_dataset() {
        let ct = CompoundTypeBuilder::new()
            .f64_field("x")
            .f64_field("y")
            .i32_field("id")
            .build();
        let elem_size = ct.type_size() as usize;
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f64.to_le_bytes());
        raw.extend_from_slice(&2.0f64.to_le_bytes());
        raw.extend_from_slice(&10i32.to_le_bytes());
        raw.extend_from_slice(&3.0f64.to_le_bytes());
        raw.extend_from_slice(&4.0f64.to_le_bytes());
        raw.extend_from_slice(&20i32.to_le_bytes());
        assert_eq!(raw.len(), 2 * elem_size);

        let mut fw = FileWriter::new();
        fw.create_dataset("particles").with_compound_data(ct, raw.clone(), 2);
        let bytes = fw.finish().unwrap();

        let (read_raw, dt) = read_dataset_raw(&bytes, "particles");
        assert_eq!(read_raw, raw);
        let fields = data_read::read_compound_fields(&read_raw, &dt).unwrap();
        assert_eq!(fields.len(), 3);
        let x_vals = data_read::read_as_f64(&fields[0].raw_data, &fields[0].datatype).unwrap();
        assert_eq!(x_vals, vec![1.0, 3.0]);
        let id_vals = data_read::read_as_i32(&fields[2].raw_data, &fields[2].datatype).unwrap();
        assert_eq!(id_vals, vec![10, 20]);
    }

    #[test]
    fn file_with_enum_dataset() {
        let et = EnumTypeBuilder::i32_based()
            .value("RED", 0)
            .value("GREEN", 1)
            .value("BLUE", 2)
            .build();

        let mut fw = FileWriter::new();
        fw.create_dataset("colors").with_enum_i32_data(et, &[1, 0, 2, 1]);
        let bytes = fw.finish().unwrap();

        let (read_raw, dt) = read_dataset_raw(&bytes, "colors");
        let names = data_read::read_enum_names(&read_raw, &dt).unwrap();
        assert_eq!(names, vec!["GREEN", "RED", "BLUE", "GREEN"]);
    }

    #[test]
    fn file_with_array_dataset() {
        // Each element is a [3] array of f64
        let mut raw = Vec::new();
        for v in &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }

        let mut fw = FileWriter::new();
        fw.create_dataset("vectors").with_array_data(
            make_f64_type(), &[3], raw.clone(), 2,
        );
        let bytes = fw.finish().unwrap();

        let (read_raw, dt) = read_dataset_raw(&bytes, "vectors");
        assert_eq!(read_raw, raw);
        let (flat, base_dt, dims) = data_read::read_array_flat(&read_raw, &dt).unwrap();
        assert_eq!(dims, vec![3]);
        let vals = data_read::read_as_f64(&flat, &base_dt).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ---- h5py round-trip tests ----

    #[cfg(feature = "std")]
    fn h5py_read(_path: &std::path::Path, script: &str) -> String {
        let o = std::process::Command::new("python3").args(["-c", script]).output().expect("python3");
        if !o.status.success() { panic!("h5py: {}", String::from_utf8_lossy(&o.stderr)); }
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_our_f64_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]).with_shape(&[3]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_f64.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['data'][:].tolist()))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_our_i32_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("ints").with_i32_data(&[10, 20, 30]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_i32.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['ints'][:].tolist()))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let values: Vec<i32> = serde_json::from_str(&stdout).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_dataset_with_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'data': d[:].tolist(), 'scale': float(d.attrs['scale'])}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["data"], serde_json::json!([1.0, 2.0]));
        assert_eq!(v["scale"], serde_json::json!(0.5));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_group_with_dataset() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
        let g = gb.finish();
        fw.add_group(g);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_grp.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); print(json.dumps(f['grp/vals'][:].tolist()))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
        assert_eq!(values, vec![10.0, 20.0]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_root_attrs() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        // Need at least a dataset for h5py to open the file
        fw.create_dataset("dummy").with_f64_data(&[0.0]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_root_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); print(int(f.attrs['version']))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        assert_eq!(stdout, "42");
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_multiple_datasets() {
        let mut fw = FileWriter::new();
        fw.create_dataset("a").with_f64_data(&[1.0]);
        fw.create_dataset("b").with_f64_data(&[2.0]);
        fw.create_dataset("c").with_f64_data(&[3.0]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_multi.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); print(json.dumps({{k: f[k][:].tolist() for k in ['a','b','c']}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["a"], serde_json::json!([1.0]));
        assert_eq!(v["b"], serde_json::json!([2.0]));
        assert_eq!(v["c"], serde_json::json!([3.0]));
    }

    // ---- Compound / Enum / Array h5py round-trips ----

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_our_compound_dataset() {
        let ct = CompoundTypeBuilder::new()
            .f64_field("x")
            .f64_field("y")
            .i32_field("id")
            .build();
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.5f64.to_le_bytes());
        raw.extend_from_slice(&2.5f64.to_le_bytes());
        raw.extend_from_slice(&10i32.to_le_bytes());
        raw.extend_from_slice(&3.5f64.to_le_bytes());
        raw.extend_from_slice(&4.5f64.to_le_bytes());
        raw.extend_from_slice(&20i32.to_le_bytes());

        let mut fw = FileWriter::new();
        fw.create_dataset("particles").with_compound_data(ct, raw, 2);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_compound.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['particles']; print(json.dumps({{'x':d['x'].tolist(),'y':d['y'].tolist(),'id':d['id'].tolist()}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["x"], serde_json::json!([1.5, 3.5]));
        assert_eq!(v["y"], serde_json::json!([2.5, 4.5]));
        assert_eq!(v["id"], serde_json::json!([10, 20]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_our_enum_dataset() {
        let et = EnumTypeBuilder::i32_based()
            .value("RED", 0)
            .value("GREEN", 1)
            .value("BLUE", 2)
            .build();

        let mut fw = FileWriter::new();
        fw.create_dataset("colors").with_enum_i32_data(et, &[1, 0, 2]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_enum.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['colors']; print(json.dumps(d[:].tolist()))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let values: Vec<i32> = serde_json::from_str(&stdout).unwrap();
        assert_eq!(values, vec![1, 0, 2]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_our_array_dataset() {
        let mut raw = Vec::new();
        for v in &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }

        let mut fw = FileWriter::new();
        fw.create_dataset("vectors").with_array_data(
            make_f64_type(), &[3], raw, 2,
        );
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_array.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['vectors']; print(json.dumps({{'shape':list(d.shape),'dtype':str(d.dtype),'values':d[:].tolist()}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["shape"], serde_json::json!([2]));
        assert_eq!(v["values"], serde_json::json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_h5py_generated_compound() {
        let path = std::env::temp_dir().join("purehdf5_h5py_compound.h5");
        let gen_script = format!(
            r#"
import h5py, numpy as np
dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('id', 'i4')])
data = np.array([(1.0, 2.0, 10), (3.0, 4.0, 20)], dtype=dt)
f = h5py.File('{}', 'w', libver='latest')
f.create_dataset('particles', data=data)
f.close()
"#,
            path.display()
        );
        h5py_read(&path, &gen_script);

        let bytes = std::fs::read(&path).unwrap();
        let (raw, dt) = read_dataset_raw(&bytes, "particles");
        let fields = data_read::read_compound_fields(&raw, &dt).unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "x");
        let x_vals = data_read::read_as_f64(&fields[0].raw_data, &fields[0].datatype).unwrap();
        assert_eq!(x_vals, vec![1.0, 3.0]);
        let id_vals = data_read::read_as_i32(&fields[2].raw_data, &fields[2].datatype).unwrap();
        assert_eq!(id_vals, vec![10, 20]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_h5py_generated_enum() {
        let path = std::env::temp_dir().join("purehdf5_h5py_enum.h5");
        let gen_script = format!(
            r#"
import h5py, numpy as np
dt = h5py.enum_dtype({{"RED": 0, "GREEN": 1, "BLUE": 2}}, basetype=np.int32)
data = np.array([1, 0, 2, 1], dtype=np.int32)
f = h5py.File('{}', 'w', libver='latest')
f.create_dataset('colors', data=data, dtype=dt)
f.close()
"#,
            path.display()
        );
        h5py_read(&path, &gen_script);

        let bytes = std::fs::read(&path).unwrap();
        let (raw, dt) = read_dataset_raw(&bytes, "colors");
        let names = data_read::read_enum_names(&raw, &dt).unwrap();
        assert_eq!(names, vec!["GREEN", "RED", "BLUE", "GREEN"]);
    }

    // Note: h5py-generated array type read test is omitted because h5py's
    // high-level API cannot write array-typed datasets. The h5py_reads_our_array_dataset
    // test above verifies the write path, and self round-trip tests cover reading.

    // ---- Chunked dataset read helper ----

    fn read_dataset_f64_any(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
        let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
        let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();

        match &dl {
            DataLayout::Chunked { .. } => {
                // Parse pipeline if present
                let pipeline = hdr.messages.iter()
                    .find(|m| m.msg_type == MessageType::FilterPipeline)
                    .map(|m| crate::filter_pipeline::FilterPipeline::parse(&m.data).unwrap());
                let raw = crate::chunked_read::read_chunked_data(
                    bytes, &dl, &ds, &dt, pipeline.as_ref(), sb.offset_size, sb.length_size,
                ).unwrap();
                data_read::read_as_f64(&raw, &dt).unwrap()
            }
            _ => {
                let raw = data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
                data_read::read_as_f64(&raw, &dt).unwrap()
            }
        }
    }

    // ---- Chunked write self round-trip tests ----

    #[test]
    fn chunked_1d_single_chunk_no_compression() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[10])
            .with_chunks(&[10]);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn chunked_1d_single_chunk_deflate() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[100])
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[test]
    fn chunked_1d_multi_chunk_no_compression() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[20])
            .with_chunks(&[8]);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn chunked_1d_multi_chunk_deflate() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[20])
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn chunked_1d_shuffle_deflate() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[50])
            .with_shuffle()
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[test]
    fn chunked_2d_no_compression() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[4, 6])
            .with_chunks(&[2, 3]);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    // ---- h5py chunked round-trip tests ----

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_no_compression() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[20]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_nocomp.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks)}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["chunks"], serde_json::json!([20]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_deflate() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[20])
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_deflate.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks),'compression':d.compression}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["chunks"], serde_json::json!([20]));
        assert_eq!(v["compression"], serde_json::json!("gzip"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_shuffle_deflate() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[50])
            .with_shuffle()
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_shuffle_deflate.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'shuffle':bool(d.shuffle),'compression':d.compression}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["shuffle"], serde_json::json!(true));
        assert_eq!(v["compression"], serde_json::json!("gzip"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_fletcher32() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[100])
            .with_chunks(&[100])
            .with_fletcher32();
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_fletcher32.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'fletcher32':bool(d.fletcher32)}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["fletcher32"], serde_json::json!(true));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_2d() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[4, 6])
            .with_chunks(&[2, 3]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_chunked_2d.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'shape':list(d.shape),'chunks':list(d.chunks),'values':d[:].flatten().tolist()}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["shape"], serde_json::json!([4, 6]));
        assert_eq!(v["chunks"], serde_json::json!([2, 3]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_2d_data() {
        let mut fw = FileWriter::new();
        fw.create_dataset("matrix")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[2, 3]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_test_2d.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py, json; f=h5py.File('{}','r'); d=f['matrix']; print(json.dumps({{'shape': list(d.shape), 'data': d[:].flatten().tolist()}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["shape"], serde_json::json!([2, 3]));
        assert_eq!(v["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    // ---- Extensible Array / resizable dataset tests ----

    #[test]
    fn resizable_dataset_self_roundtrip() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[20])
            .with_chunks(&[10])
            .with_maxshape(&[u64::MAX]);
        let bytes = fw.finish().unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        assert_eq!(result, data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_resizable_dataset() {
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[50])
            .with_chunks(&[10])
            .with_maxshape(&[u64::MAX]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_ea_resizable.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'chunks':list(d.chunks),'maxshape':list(None if x is None else x for x in d.maxshape)}}))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["chunks"], serde_json::json!([10]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_h5py_generated_ea_file() {
        // Generate an EA file with h5py, then read it with Rust
        let path = std::env::temp_dir().join("purehdf5_h5py_ea.h5");
        let gen_script = format!(
            "import h5py,numpy as np; f=h5py.File('{}','w'); d=f.create_dataset('data',data=np.arange(30,dtype='float64'),chunks=(10,),maxshape=(None,)); f.close()",
            path.display()
        );
        h5py_read(&path, &gen_script);

        let bytes = std::fs::read(&path).unwrap();
        let result = read_dataset_f64_any(&bytes, "data");
        let expected: Vec<f64> = (0..30).map(|i| i as f64).collect();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_append_and_verify() {
        // This is the PRD gate test:
        // 1. Create a resizable dataset with Rust
        // 2. Use h5py to append data 3 times
        // 3. Verify the final data with h5py
        let mut fw = FileWriter::new();
        let initial: Vec<f64> = (0..10).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&initial)
            .with_shape(&[10])
            .with_chunks(&[10])
            .with_maxshape(&[u64::MAX]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_ea_append.h5");
        std::fs::write(&path, &bytes).unwrap();

        // Append 3 times using h5py and verify final data
        let script = format!(
            r#"
import h5py, json, numpy as np

# Read initial data
f = h5py.File('{}', 'a')
d = f['data']

# Append 3 batches
for batch in range(3):
    old_size = d.shape[0]
    new_data = np.arange(old_size, old_size + 10, dtype='float64')
    d.resize(old_size + 10, axis=0)
    d[old_size:] = new_data

f.close()

# Re-read and verify
f = h5py.File('{}', 'r')
d = f['data']
result = d[:].tolist()
shape = list(d.shape)
f.close()

print(json.dumps({{'values': result, 'shape': shape}}))
"#,
            path.display(),
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        let expected: Vec<f64> = (0..40).map(|i| i as f64).collect();
        assert_eq!(values, expected);
        assert_eq!(v["shape"], serde_json::json!([40]));

        // Also verify we can read the appended file back with Rust
        let final_bytes = std::fs::read(&path).unwrap();
        let final_result = read_dataset_f64_any(&final_bytes, "data");
        assert_eq!(final_result, expected);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_resizable_single_chunk() {
        // Single chunk resizable dataset
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[5])
            .with_chunks(&[10])
            .with_maxshape(&[u64::MAX]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("purehdf5_ea_single.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps(d[:].tolist()))",
            path.display()
        );
        let stdout = h5py_read(&path, &script);
        let values: Vec<f64> = serde_json::from_str(&stdout).unwrap();
        assert_eq!(values, data);
    }
}
