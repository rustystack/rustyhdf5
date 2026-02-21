//! Memory-mapped HDF5 file reader.
//!
//! [`MmapFile`] provides the same API as [`crate::File`] but uses memory-mapped
//! I/O instead of reading the entire file into a `Vec<u8>`.  For contiguous
//! datasets the read path is **zero-copy** — callers receive a slice directly
//! into the mmap region.

use std::collections::HashMap;

use purehdf5_format::attribute::extract_attributes_full;
use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read;
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
use purehdf5_format::error::FormatError;
use purehdf5_format::filter_pipeline::FilterPipeline;
use purehdf5_format::group_v1::{self, GroupEntry};
use purehdf5_format::group_v2;
use purehdf5_format::message_type::MessageType;
use purehdf5_format::object_header::ObjectHeader;
use purehdf5_format::signature;
use purehdf5_format::superblock::Superblock;
use purehdf5_format::symbol_table::SymbolTableMessage;

use purehdf5_io::MmapReader;

use crate::error::Error;
use crate::types::{attrs_to_map, classify_datatype, AttrValue, DType};

/// An HDF5 file opened via memory mapping.
///
/// Uses `MmapReader` under the hood so every access is a zero-copy view into
/// the kernel page cache.  For contiguous datasets you can obtain a direct
/// `&[u8]` slice via [`MmapDataset::read_raw_slice`].
pub struct MmapFile {
    reader: MmapReader,
    superblock: Superblock,
}

impl MmapFile {
    /// Open an HDF5 file using memory-mapped I/O.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let reader = MmapReader::open(path).map_err(Error::Io)?;
        let data = reader.as_bytes();
        let sig_offset = signature::find_signature(data)?;
        let superblock = Superblock::parse(data, sig_offset)?;
        Ok(Self { reader, superblock })
    }

    /// Returns a handle to the root group.
    pub fn root(&self) -> MmapGroup<'_> {
        MmapGroup {
            file: self,
            address: self.superblock.root_group_address,
        }
    }

    /// Resolve a path and return a `MmapDataset` handle.
    pub fn dataset(&self, path: &str) -> Result<MmapDataset<'_>, Error> {
        let data = self.reader.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        let hdr = self.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(MmapDataset { file: self, header: hdr })
    }

    /// Resolve a path and return a `MmapGroup` handle.
    pub fn group(&self, path: &str) -> Result<MmapGroup<'_>, Error> {
        let data = self.reader.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        Ok(MmapGroup { file: self, address: addr })
    }

    /// Returns the raw file bytes (zero-copy from mmap).
    pub fn as_bytes(&self) -> &[u8] {
        self.reader.as_bytes()
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        ObjectHeader::parse(
            self.reader.as_bytes(),
            address as usize,
            self.superblock.offset_size,
            self.superblock.length_size,
        )
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
}

impl std::fmt::Debug for MmapFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapFile")
            .field("size", &self.reader.len())
            .field("superblock_version", &self.superblock.version)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 group inside an [`MmapFile`].
pub struct MmapGroup<'f> {
    file: &'f MmapFile,
    address: u64,
}

impl<'f> MmapGroup<'f> {
    /// List the names of datasets in this group.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// List the names of subgroups in this group.
    pub fn groups(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let data = self.file.reader.as_bytes();
        let hdr = self.file.parse_header(self.address)?;
        let attr_msgs = extract_attributes_full(
            data,
            &hdr,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            data,
            self.file.offset_size(),
            self.file.length_size(),
        ))
    }

    /// Get a dataset within this group by name.
    pub fn dataset(&self, name: &str) -> Result<MmapDataset<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        Ok(MmapDataset { file: self.file, header: hdr })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<MmapGroup<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(MmapGroup {
            file: self.file,
            address: entry.object_header_address,
        })
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let data = self.file.reader.as_bytes();
        let hdr = self.file.parse_header(self.address)?;
        let os = self.file.offset_size();
        let ls = self.file.length_size();
        resolve_group_entries(data, &hdr, os, ls).map_err(Error::Format)
    }
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 dataset inside an [`MmapFile`].
#[derive(Debug)]
pub struct MmapDataset<'f> {
    file: &'f MmapFile,
    header: ObjectHeader,
}

impl<'f> MmapDataset<'f> {
    /// Returns the shape (dimensions) of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        let ds = self.dataspace()?;
        Ok(ds.dimensions.clone())
    }

    /// Returns the simplified datatype of the dataset.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
    }

    /// Read all data as `f64` values.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64(&raw, &dt)?)
    }

    /// Zero-copy read of contiguous native-endian `f64` data.
    ///
    /// Returns `Some(&[f64])` when the dataset is contiguous and stored as
    /// little-endian `f64` with proper alignment.  Returns `None` for
    /// chunked/compact layouts or non-native types — use [`read_f64`] instead.
    pub fn read_f64_zerocopy(&self) -> Result<Option<&'f [f64]>, Error> {
        let raw = match self.read_raw_slice()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64_zerocopy(raw, &dt))
    }

    /// Zero-copy read of contiguous native-endian `f32` data.
    ///
    /// Returns `Some(&[f32])` when the dataset is contiguous and stored as
    /// little-endian `f32` with proper alignment.  Returns `None` otherwise.
    pub fn read_f32_zerocopy(&self) -> Result<Option<&'f [f32]>, Error> {
        let raw = match self.read_raw_slice()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32_zerocopy(raw, &dt))
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32(&raw, &dt)?)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i32(&raw, &dt)?)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i64(&raw, &dt)?)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u64(&raw, &dt)?)
    }

    /// Read all data as `String` values.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_strings(&raw, &dt)?)
    }

    /// For contiguous datasets, return a zero-copy slice into the mmap.
    ///
    /// Returns `None` if the dataset is not contiguous (compact or chunked).
    pub fn read_raw_slice(&self) -> Result<Option<&'f [u8]>, Error> {
        let dl = self.data_layout()?;
        let ds = self.dataspace()?;
        let dt = self.datatype()?;
        let expected = ds.num_elements() as usize * dt.type_size() as usize;
        match &dl {
            DataLayout::Contiguous { address, size } => {
                let addr = address.ok_or(Error::Format(FormatError::NoDataAllocated))?;
                let sz = *size as usize;
                if sz != expected {
                    return Err(Error::Format(FormatError::DataSizeMismatch {
                        expected,
                        actual: sz,
                    }));
                }
                let data = self.file.reader.as_bytes();
                let a = addr as usize;
                if a + sz > data.len() {
                    return Err(Error::Format(FormatError::UnexpectedEof {
                        expected: a + sz,
                        available: data.len(),
                    }));
                }
                Ok(Some(&data[a..a + sz]))
            }
            _ => Ok(None),
        }
    }

    /// Read all attributes of this dataset.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let data = self.file.reader.as_bytes();
        let attr_msgs = extract_attributes_full(
            data,
            &self.header,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            data,
            self.file.offset_size(),
            self.file.length_size(),
        ))
    }

    fn datatype(&self) -> Result<Datatype, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&msg.data)?;
        Ok(dt)
    }

    fn dataspace(&self) -> Result<Dataspace, Error> {
        let msg = find_message(&self.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(&msg.data, self.file.length_size())?)
    }

    fn data_layout(&self) -> Result<DataLayout, Error> {
        let msg = find_message(&self.header, MessageType::DataLayout)?;
        Ok(DataLayout::parse(
            &msg.data,
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }

    fn filter_pipeline(&self) -> Option<FilterPipeline> {
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok())
    }

    fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.data_layout()?;
        let pipeline = self.filter_pipeline();
        Ok(data_read::read_raw_data_full(
            self.file.reader.as_bytes(),
            &dl,
            &ds,
            &dt,
            pipeline.as_ref(),
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }
}

// ---------------------------------------------------------------------------
// Helpers (same as reader.rs)
// ---------------------------------------------------------------------------

fn find_message(
    header: &ObjectHeader,
    msg_type: MessageType,
) -> Result<&purehdf5_format::object_header::HeaderMessage, Error> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == msg_type)
        .ok_or(Error::MissingMessage(msg_type))
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
}

fn is_group(header: &ObjectHeader) -> bool {
    header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LinkInfo
            || m.msg_type == MessageType::Link
            || m.msg_type == MessageType::SymbolTable)
}

fn resolve_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    let is_v1 = object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::SymbolTable);
    let is_v2 = object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LinkInfo || m.msg_type == MessageType::Link);

    if is_v1 {
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .ok_or_else(|| FormatError::PathNotFound("no symbol table message".into()))?;
        let stm = SymbolTableMessage::parse(&sym_msg.data, offset_size)?;
        group_v1::resolve_v1_group_entries(file_data, &stm, offset_size, length_size)
    } else if is_v2 {
        group_v2::resolve_v2_group_entries(file_data, object_header, offset_size, length_size)
    } else {
        Ok(Vec::new())
    }
}
