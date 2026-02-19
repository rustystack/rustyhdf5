//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, and inline attributes.

use crate::attribute::AttributeMessage;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{CharacterSet, Datatype, DatatypeByteOrder, StringPadding};
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
    data: Option<Vec<u8>>,
    attrs: Vec<(String, AttrValue)>,
}

impl DatasetBuilder {
    fn new(name: &str) -> Self {
        Self { name: name.to_string(), datatype: None, shape: None, data: None, attrs: Vec::new() }
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

    pub fn with_shape(&mut self, shape: &[u64]) -> &mut Self {
        self.shape = Some(shape.to_vec());
        self
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.attrs.push((name.to_string(), value));
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
        // Key insight: dataset OH size is independent of data address (always 8 bytes).
        // So we can compute sizes with a dummy address, assign real addresses, then rebuild.

        // 1. Extract structured info from builders
        struct DsFlat {
            name: String,
            dt: Datatype,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            _parent: usize, // 0=root, 1..=N = group index+1
        }
        struct GrpFlat {
            name: String,
            attrs: Vec<AttributeMessage>,
            ds_indices: Vec<usize>,
        }

        let mut all_ds: Vec<DsFlat> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();

        // Root datasets
        let mut root_ds_indices: Vec<usize> = Vec::new();
        for db in self.root_datasets {
            let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
            let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
            let raw = db.data.ok_or(FormatError::DatasetMissingData)?;
            let dspace = Dataspace {
                space_type: if shape.is_empty() { DataspaceType::Scalar } else { DataspaceType::Simple },
                rank: shape.len() as u8, dimensions: shape, max_dimensions: None,
            };
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs { attrs.push(build_attr_message(n, v)); }
            let idx = all_ds.len();
            root_ds_indices.push(idx);
            all_ds.push(DsFlat { name: db.name, dt, ds: dspace, raw, attrs, _parent: 0 });
        }

        // Groups + their datasets
        for (gi, g) in self.groups.into_iter().enumerate() {
            let mut gattrs = Vec::new();
            for (n, v) in &g.attrs { gattrs.push(build_attr_message(n, v)); }
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
                let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
                let raw = db.data.ok_or(FormatError::DatasetMissingData)?;
                let dspace = Dataspace {
                    space_type: if shape.is_empty() { DataspaceType::Scalar } else { DataspaceType::Simple },
                    rank: shape.len() as u8, dimensions: shape, max_dimensions: None,
                };
                let mut attrs = Vec::new();
                for (n, v) in &db.attrs { attrs.push(build_attr_message(n, v)); }
                let idx = all_ds.len();
                ds_idx.push(idx);
                all_ds.push(DsFlat { name: db.name, dt, ds: dspace, raw, attrs, _parent: gi + 1 });
            }
            groups.push(GrpFlat { name: g.name, attrs: gattrs, ds_indices: ds_idx });
        }

        // Root attrs
        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        for (n, v) in &self.root_attrs { root_attrs.push(build_attr_message(n, v)); }

        // 2. Compute OH sizes with dummy addresses (data_addr=0)
        let ds_oh_sizes: Vec<usize> = all_ds.iter().map(|d| {
            build_dataset_oh(&d.dt, &d.ds, 0, d.raw.len() as u64, &d.attrs).len()
        }).collect();

        // Group OH sizes depend on link messages, which depend on child addresses.
        // But link message size doesn't depend on the address value (always 8 bytes for offset_size=8).
        // So group OH size is also stable.
        let group_oh_sizes: Vec<usize> = groups.iter().map(|g| {
            let dummy_links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| {
                make_link(&all_ds[i].name, 0)
            }).collect();
            build_group_oh(&dummy_links, &g.attrs).len()
        }).collect();

        // Root group: links to root datasets + links to groups
        let root_dummy_links: Vec<LinkMessage> = {
            let mut links = Vec::new();
            for &i in &root_ds_indices { links.push(make_link(&all_ds[i].name, 0)); }
            for g in &groups { links.push(make_link(&g.name, 0)); }
            links
        };
        let root_oh_size = build_group_oh(&root_dummy_links, &root_attrs).len();

        // 3. Assign addresses
        let root_group_addr = SUPERBLOCK_SIZE as u64;
        let mut cursor = SUPERBLOCK_SIZE + root_oh_size;

        // Group OH addresses
        let group_addrs: Vec<u64> = group_oh_sizes.iter().map(|&sz| {
            let addr = cursor as u64;
            cursor += sz;
            addr
        }).collect();

        // Dataset OH addresses
        let ds_oh_addrs: Vec<u64> = ds_oh_sizes.iter().map(|&sz| {
            let addr = cursor as u64;
            cursor += sz;
            addr
        }).collect();

        // Raw data addresses
        let ds_data_addrs: Vec<u64> = all_ds.iter().map(|d| {
            let addr = cursor as u64;
            cursor += d.raw.len();
            addr
        }).collect();

        let eof_addr = cursor as u64;

        // 4. Build actual bytes
        let mut buf = Vec::with_capacity(cursor);

        // Superblock
        let sb = Superblock {
            version: 3,
            offset_size: OFFSET_SIZE,
            length_size: LENGTH_SIZE,
            base_address: 0,
            eof_address: eof_addr,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(u64::MAX),
            checksum: None, // computed by serialize()
        };
        buf.extend_from_slice(&sb.serialize());

        // Root group OH
        let mut root_links: Vec<LinkMessage> = Vec::new();
        for &i in &root_ds_indices {
            root_links.push(make_link(&all_ds[i].name, ds_oh_addrs[i]));
        }
        for (gi, g) in groups.iter().enumerate() {
            root_links.push(make_link(&g.name, group_addrs[gi]));
        }
        buf.extend_from_slice(&build_group_oh(&root_links, &root_attrs));

        // Group OHs
        for g in &groups {
            let links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| {
                make_link(&all_ds[i].name, ds_oh_addrs[i])
            }).collect();
            buf.extend_from_slice(&build_group_oh(&links, &g.attrs));
        }

        // Dataset OHs
        for (i, d) in all_ds.iter().enumerate() {
            buf.extend_from_slice(&build_dataset_oh(
                &d.dt, &d.ds, ds_data_addrs[i], d.raw.len() as u64, &d.attrs,
            ));
        }

        // Raw data
        for d in &all_ds {
            buf.extend_from_slice(&d.raw);
        }

        debug_assert_eq!(buf.len(), cursor);
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

    // ---- h5py round-trip tests ----

    #[cfg(feature = "std")]
    fn h5py_read(path: &std::path::Path, script: &str) -> String {
        let output = std::process::Command::new("python3")
            .args(["-c", script])
            .output()
            .expect("python3 not found");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("h5py script failed: {stderr}");
        }
        String::from_utf8(output.stdout).unwrap().trim().to_string()
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
}
