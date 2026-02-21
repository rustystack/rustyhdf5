//! I/O abstraction layer for HDF5 file access.
//!
//! Provides traits and adapters for reading and writing HDF5 data
//! from files, memory buffers, and optionally memory-mapped files.

use std::io::{self, Read, Seek, SeekFrom, Write};

pub use purehdf5_format;

/// Read-only access to HDF5 data.
///
/// Implementors provide the ability to read the entire file content
/// as a byte slice, which is the interface that `purehdf5-format` expects.
pub trait HDF5Read {
    /// Returns the entire file content as a byte slice.
    fn as_bytes(&self) -> &[u8];

    /// Returns the length of the data in bytes.
    fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns true if the data is empty.
    fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
}

/// Read-write access to HDF5 data.
///
/// Implementors can both read existing data and write new data.
pub trait HDF5ReadWrite: HDF5Read {
    /// Write the given bytes to the underlying storage, replacing all content.
    fn write_all_bytes(&mut self, data: &[u8]) -> io::Result<()>;
}

// ---------------------------------------------------------------------------
// MemoryReader — wraps a Vec<u8> or borrowed &[u8] for in-memory access
// ---------------------------------------------------------------------------

/// In-memory reader backed by an owned `Vec<u8>`.
///
/// This mirrors what `purehdf5-format` currently does: the entire file
/// is held in memory as a byte vector.
#[derive(Debug, Clone)]
pub struct MemoryReader {
    data: Vec<u8>,
}

impl MemoryReader {
    /// Create a reader from an owned byte vector.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create a reader by copying from a byte slice.
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Consume the reader and return the underlying bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

impl HDF5Read for MemoryReader {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl HDF5ReadWrite for MemoryReader {
    fn write_all_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        self.data = data.to_vec();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BorrowedReader — wraps &[u8] without copying
// ---------------------------------------------------------------------------

/// Zero-copy reader over a borrowed byte slice.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedReader<'a> {
    data: &'a [u8],
}

impl<'a> BorrowedReader<'a> {
    /// Create a reader from a borrowed byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl HDF5Read for BorrowedReader<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// FileReader — wraps std::fs::File for read access
// ---------------------------------------------------------------------------

/// File-backed reader that loads the entire file into memory.
///
/// Uses `Read + Seek` to slurp the file content into a `Vec<u8>`.
#[derive(Debug)]
pub struct FileReader {
    data: Vec<u8>,
}

impl FileReader {
    /// Open a file and read its entire contents into memory.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let len = file.seek(SeekFrom::End(0))? as usize;
        file.seek(SeekFrom::Start(0))?;
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;
        Ok(Self { data })
    }

    /// Create a reader from an already-opened file.
    pub fn from_file(mut file: std::fs::File) -> io::Result<Self> {
        let len = file.seek(SeekFrom::End(0))? as usize;
        file.seek(SeekFrom::Start(0))?;
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;
        Ok(Self { data })
    }

    /// Consume the reader and return the underlying bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

impl HDF5Read for FileReader {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// FileWriter — wraps std::fs::File for write access
// ---------------------------------------------------------------------------

/// File-backed writer that writes bytes to a file on disk.
#[derive(Debug)]
pub struct FileWriter {
    path: std::path::PathBuf,
    data: Vec<u8>,
}

impl FileWriter {
    /// Create a new writer that will write to the given path.
    pub fn create<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        Ok(Self {
            path,
            data: Vec::new(),
        })
    }

    /// Flush the current data to disk.
    pub fn flush_to_disk(&self) -> io::Result<()> {
        let mut file = std::fs::File::create(&self.path)?;
        file.write_all(&self.data)?;
        file.flush()
    }

    /// Returns the target path.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl HDF5Read for FileWriter {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl HDF5ReadWrite for FileWriter {
    fn write_all_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        self.data = data.to_vec();
        self.flush_to_disk()
    }
}

// ---------------------------------------------------------------------------
// Optional modules
// ---------------------------------------------------------------------------

#[cfg(feature = "async")]
pub mod async_read;

#[cfg(feature = "hsds")]
pub mod hsds;

#[cfg(feature = "mmap")]
pub mod mmap;

#[cfg(feature = "mmap")]
pub use mmap::{MmapReader, MmapReadWrite};

pub mod prefetch;
pub mod sweep;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn memory_reader_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5];
        let reader = MemoryReader::new(data.clone());
        assert_eq!(reader.as_bytes(), &data);
        assert_eq!(reader.len(), 5);
        assert!(!reader.is_empty());
    }

    #[test]
    fn memory_reader_from_slice() {
        let data = [10u8, 20, 30];
        let reader = MemoryReader::from_slice(&data);
        assert_eq!(reader.as_bytes(), &data);
    }

    #[test]
    fn memory_reader_empty() {
        let reader = MemoryReader::new(Vec::new());
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn memory_reader_into_inner() {
        let data = vec![7u8, 8, 9];
        let reader = MemoryReader::new(data.clone());
        assert_eq!(reader.into_inner(), data);
    }

    #[test]
    fn memory_reader_write_replaces_content() {
        let mut reader = MemoryReader::new(vec![1, 2, 3]);
        reader.write_all_bytes(&[4, 5]).unwrap();
        assert_eq!(reader.as_bytes(), &[4, 5]);
    }

    #[test]
    fn borrowed_reader_basic() {
        let data = [42u8, 43, 44];
        let reader = BorrowedReader::new(&data);
        assert_eq!(reader.as_bytes(), &data);
        assert_eq!(reader.len(), 3);
        assert!(!reader.is_empty());
    }

    #[test]
    fn borrowed_reader_empty() {
        let reader = BorrowedReader::new(&[]);
        assert!(reader.is_empty());
    }

    #[test]
    fn file_reader_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_file_reader.bin");

        // Write test data
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0x89, 0x48, 0x44, 0x46]).unwrap();
        }

        let reader = FileReader::open(&path).unwrap();
        assert_eq!(reader.as_bytes(), &[0x89, 0x48, 0x44, 0x46]);
        assert_eq!(reader.len(), 4);

        let bytes = reader.into_inner();
        assert_eq!(bytes, vec![0x89, 0x48, 0x44, 0x46]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_reader_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_from_file.bin");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[1, 2, 3, 4, 5, 6]).unwrap();
        }

        let file = std::fs::File::open(&path).unwrap();
        let reader = FileReader::from_file(file).unwrap();
        assert_eq!(reader.as_bytes(), &[1, 2, 3, 4, 5, 6]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_reader_nonexistent() {
        let result = FileReader::open("/tmp/purehdf5_io_does_not_exist_12345.bin");
        assert!(result.is_err());
    }

    #[test]
    fn file_writer_create_and_write() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_writer.bin");

        let mut writer = FileWriter::create(&path).unwrap();
        assert!(writer.as_bytes().is_empty());

        writer.write_all_bytes(&[10, 20, 30]).unwrap();
        assert_eq!(writer.as_bytes(), &[10, 20, 30]);

        // Verify the file was written to disk
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk, vec![10, 20, 30]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_writer_overwrite() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_writer_overwrite.bin");

        let mut writer = FileWriter::create(&path).unwrap();
        writer.write_all_bytes(&[1, 2, 3]).unwrap();
        writer.write_all_bytes(&[4, 5, 6, 7]).unwrap();

        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk, vec![4, 5, 6, 7]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_writer_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_path.bin");
        let writer = FileWriter::create(&path).unwrap();
        assert_eq!(writer.path(), path.as_path());
    }

    #[test]
    fn file_writer_flush_to_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_flush.bin");

        let mut writer = FileWriter::create(&path).unwrap();
        writer.write_all_bytes(&[0xDE, 0xAD]).unwrap();
        writer.flush_to_disk().unwrap();

        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk, vec![0xDE, 0xAD]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn hdf5_file_via_memory_reader() {
        // Integration test: use MemoryReader with an HDF5 file created by purehdf5-format
        use purehdf5_format::file_writer::FileWriter as FmtWriter;

        let mut fw = FmtWriter::new();
        fw.create_dataset("test").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();

        let reader = MemoryReader::new(bytes);
        let data = reader.as_bytes();

        // Verify it's a valid HDF5 file by checking the signature
        assert!(data.len() > 8);
        assert_eq!(&data[..8], b"\x89HDF\r\n\x1a\n");
    }

    #[test]
    fn hdf5_file_via_file_reader_writer() {
        use purehdf5_format::file_writer::FileWriter as FmtWriter;

        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_io_test_hdf5_roundtrip.h5");

        // Write an HDF5 file via FileWriter
        let mut fw = FmtWriter::new();
        fw.create_dataset("values").with_i32_data(&[10, 20, 30]);
        let bytes = fw.finish().unwrap();

        let mut writer = FileWriter::create(&path).unwrap();
        writer.write_all_bytes(&bytes).unwrap();

        // Read it back via FileReader
        let reader = FileReader::open(&path).unwrap();
        assert_eq!(reader.as_bytes(), &bytes);
        assert_eq!(&reader.as_bytes()[..8], b"\x89HDF\r\n\x1a\n");

        std::fs::remove_file(&path).ok();
    }
}
