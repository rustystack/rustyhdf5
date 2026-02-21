//! Lazy file handle that only parses HDF5 metadata on demand.
//!
//! Unlike [`crate::File`] which loads the entire file into a `Vec<u8>`,
//! [`LazyFile`] works with any [`purehdf5_io::HDF5Read`] backend (including
//! memory-mapped files) and only parses metadata as needed:
//!
//! - On open: parse ONLY superblock + root group object header
//! - On dataset access: parse dataset's object header on demand, cache result
//! - On attribute access: parse attributes on demand
//! - On data read: parse data layout + read data (no caching of raw data)

use std::cell::RefCell;
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

use purehdf5_io::HDF5Read;

use crate::error::Error;
use crate::types::{attrs_to_map, classify_datatype, AttrValue, DType};

/// A lazy HDF5 file handle that parses metadata on demand.
///
/// On open, only the superblock and root group object header are parsed.
/// Subsequent accesses to datasets, groups, and attributes trigger parsing
/// of just the needed metadata, which is cached for future access.
///
/// Works with any [`HDF5Read`] backend: `FileReader`, `MmapReader`,
/// `MemoryReader`, etc.
pub struct LazyFile<R: HDF5Read> {
    reader: R,
    superblock: Superblock,
    root_header: ObjectHeader,
    /// Cache of parsed object headers, keyed by address.
    header_cache: RefCell<HashMap<u64, ObjectHeader>>,
}

impl LazyFile<purehdf5_io::MemoryReader> {
    /// Open a lazy file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        let reader = purehdf5_io::MemoryReader::new(data);
        Self::open(reader)
    }
}

#[cfg(feature = "mmap")]
impl LazyFile<purehdf5_io::MmapReader> {
    /// Open a lazy file using memory-mapped I/O.
    ///
    /// This is the recommended way to open large files: only the superblock
    /// and root group header are parsed eagerly; everything else is on-demand.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let reader = purehdf5_io::MmapReader::open(path).map_err(Error::Io)?;
        Self::open(reader)
    }
}

impl<R: HDF5Read> LazyFile<R> {
    /// Open a lazy file from any `HDF5Read` backend.
    ///
    /// Parses only the superblock and root group object header.
    pub fn open(reader: R) -> Result<Self, Error> {
        let data = reader.as_bytes();
        let sig_offset = signature::find_signature(data)?;
        let superblock = Superblock::parse(data, sig_offset)?;
        let root_header = ObjectHeader::parse(
            data,
            superblock.root_group_address as usize,
            superblock.offset_size,
            superblock.length_size,
        )?;
        Ok(Self {
            reader,
            superblock,
            root_header,
            header_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Returns the raw file bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.reader.as_bytes()
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Returns a handle to the root group.
    pub fn root(&self) -> LazyGroup<'_, R> {
        LazyGroup {
            file: self,
            address: self.superblock.root_group_address,
        }
    }

    /// Resolve a path and return a `LazyDataset` handle.
    pub fn dataset(&self, path: &str) -> Result<LazyDataset<'_, R>, Error> {
        let data = self.reader.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        let hdr = self.get_or_parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(LazyDataset {
            file: self,
            header: hdr,
        })
    }

    /// Resolve a path and return a `LazyGroup` handle.
    pub fn group(&self, path: &str) -> Result<LazyGroup<'_, R>, Error> {
        let data = self.reader.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        Ok(LazyGroup {
            file: self,
            address: addr,
        })
    }

    /// Returns the number of cached headers.
    pub fn cached_header_count(&self) -> usize {
        self.header_cache.borrow().len()
    }

    /// Returns the root group object header (parsed on open).
    pub fn root_header(&self) -> &ObjectHeader {
        &self.root_header
    }

    /// Access the inner reader.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Parse or retrieve from cache an object header at the given address.
    fn get_or_parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        // Check if the root header matches
        if address == self.superblock.root_group_address {
            return Ok(self.root_header.clone());
        }

        // Check cache
        {
            let cache = self.header_cache.borrow();
            if let Some(hdr) = cache.get(&address) {
                return Ok(hdr.clone());
            }
        }

        // Parse and cache
        let data = self.reader.as_bytes();
        let hdr = ObjectHeader::parse(
            data,
            address as usize,
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;

        self.header_cache
            .borrow_mut()
            .insert(address, hdr.clone());
        Ok(hdr)
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
}

impl<R: HDF5Read> std::fmt::Debug for LazyFile<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFile")
            .field("size", &self.reader.as_bytes().len())
            .field("superblock_version", &self.superblock.version)
            .field("cached_headers", &self.header_cache.borrow().len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// LazyGroup handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 group in a [`LazyFile`].
pub struct LazyGroup<'f, R: HDF5Read> {
    file: &'f LazyFile<R>,
    address: u64,
}

impl<'f, R: HDF5Read> LazyGroup<'f, R> {
    /// List the names of datasets in this group.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.get_or_parse_header(entry.object_header_address)?;
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
            let hdr = self.file.get_or_parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let hdr = self.file.get_or_parse_header(self.address)?;
        let data = self.file.reader.as_bytes();
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
    pub fn dataset(&self, name: &str) -> Result<LazyDataset<'f, R>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.get_or_parse_header(entry.object_header_address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        Ok(LazyDataset {
            file: self.file,
            header: hdr,
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<LazyGroup<'f, R>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(LazyGroup {
            file: self.file,
            address: entry.object_header_address,
        })
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let hdr = self.file.get_or_parse_header(self.address)?;
        let data = self.file.reader.as_bytes();
        let os = self.file.offset_size();
        let ls = self.file.length_size();
        resolve_group_entries(data, &hdr, os, ls).map_err(Error::Format)
    }
}

// ---------------------------------------------------------------------------
// LazyDataset handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 dataset in a [`LazyFile`].
#[derive(Debug)]
pub struct LazyDataset<'f, R: HDF5Read> {
    file: &'f LazyFile<R>,
    header: ObjectHeader,
}

impl<'f, R: HDF5Read> LazyDataset<'f, R> {
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
        let raw = match self.read_raw_ref()? {
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
        let raw = match self.read_raw_ref()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32_zerocopy(raw, &dt))
    }

    /// Zero-copy read of contiguous raw data.
    ///
    /// For contiguous datasets this returns a direct `&[u8]` slice into
    /// the underlying reader bytes — no allocation.
    /// Returns `None` for chunked or compact datasets.
    pub fn read_raw_ref(&self) -> Result<Option<&'f [u8]>, Error> {
        let dl = self.data_layout()?;
        let ds = self.dataspace()?;
        let dt = self.datatype()?;
        let slice = data_read::read_raw_data_zerocopy(
            self.file.reader.as_bytes(),
            &dl,
            &ds,
            &dt,
        )?;
        Ok(slice)
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
        let data = self.file.reader.as_bytes();
        Ok(data_read::read_raw_data_full(
            data,
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
    header.messages.iter().any(|m| {
        m.msg_type == MessageType::LinkInfo
            || m.msg_type == MessageType::Link
            || m.msg_type == MessageType::SymbolTable
    })
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
