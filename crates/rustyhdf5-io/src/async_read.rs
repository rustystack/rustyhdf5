//! Async I/O support for HDF5 file access.
//!
//! Provides async traits and adapters for reading HDF5 data using tokio.
//! Gated behind the `async` feature flag.

use std::io;
use std::path::Path;

use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::data_read::{
    read_as_f32, read_as_f64, read_as_i32, read_as_i64, read_as_u64, read_raw_data_full,
};
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::error::FormatError;
use rustyhdf5_format::filter_pipeline::FilterPipeline;
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::object_header::{HeaderMessage, ObjectHeader};
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;
use tokio::io::AsyncReadExt;

/// Async read-only access to HDF5 data.
///
/// This is the async counterpart to [`super::HDF5Read`]. Implementations
/// load bytes asynchronously and then expose them for parsing.
pub trait AsyncHDF5Read: Send + Sync {
    /// Read data from the source at the given offset.
    ///
    /// Returns the bytes read. The length of the returned slice may be
    /// less than `len` if the source is shorter.
    fn read_at(
        &self,
        offset: u64,
        len: usize,
    ) -> impl std::future::Future<Output = io::Result<Vec<u8>>> + Send;

    /// Returns the total length of the underlying data source in bytes.
    fn len(&self) -> impl std::future::Future<Output = io::Result<u64>> + Send;

    /// Returns true if the underlying data source is empty.
    fn is_empty(&self) -> impl std::future::Future<Output = io::Result<bool>> + Send {
        async { Ok(self.len().await? == 0) }
    }

    /// Read the entire source into memory.
    fn read_all(&self) -> impl std::future::Future<Output = io::Result<Vec<u8>>> + Send {
        async {
            let total = self.len().await?;
            self.read_at(0, total as usize).await
        }
    }
}

// ---------------------------------------------------------------------------
// AsyncFileReader — wraps tokio::fs::File for async access
// ---------------------------------------------------------------------------

/// Async file-backed reader using tokio for non-blocking I/O.
///
/// Opens a file and reads it asynchronously. The file is read into memory
/// on first access, making subsequent operations fast.
#[derive(Debug)]
pub struct AsyncFileReader {
    path: std::path::PathBuf,
}

impl AsyncFileReader {
    /// Create a new async file reader for the given path.
    ///
    /// The file is not opened until a read method is called.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Open a file and read its entire contents asynchronously.
    pub async fn open<P: AsRef<Path>>(path: P) -> io::Result<AsyncFileReaderLoaded> {
        let mut file = tokio::fs::File::open(path.as_ref()).await?;
        let metadata = file.metadata().await?;
        let len = metadata.len();
        let mut data = Vec::with_capacity(len as usize);
        file.read_to_end(&mut data).await?;
        Ok(AsyncFileReaderLoaded { data })
    }
}

impl AsyncHDF5Read for AsyncFileReader {
    async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut file = tokio::fs::File::open(&self.path).await?;
        let metadata = file.metadata().await?;
        let file_len = metadata.len();
        if offset >= file_len {
            return Ok(Vec::new());
        }
        let available = (file_len - offset) as usize;
        let to_read = len.min(available);
        tokio::io::AsyncSeekExt::seek(&mut file, io::SeekFrom::Start(offset)).await?;
        let mut buf = vec![0u8; to_read];
        file.read_exact(&mut buf).await?;
        Ok(buf)
    }

    async fn len(&self) -> io::Result<u64> {
        let metadata = tokio::fs::metadata(&self.path).await?;
        Ok(metadata.len())
    }
}

/// A fully-loaded async file reader (file contents already in memory).
#[derive(Debug, Clone)]
pub struct AsyncFileReaderLoaded {
    data: Vec<u8>,
}

impl AsyncFileReaderLoaded {
    /// Access the loaded bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume and return the underlying bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

impl AsyncHDF5Read for AsyncFileReaderLoaded {
    async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let off = offset as usize;
        if off >= self.data.len() {
            return Ok(Vec::new());
        }
        let end = (off + len).min(self.data.len());
        Ok(self.data[off..end].to_vec())
    }

    async fn len(&self) -> io::Result<u64> {
        Ok(self.data.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// AsyncMemoryReader — wraps bytes for async-consistent API
// ---------------------------------------------------------------------------

/// In-memory async reader for consistent API across sync and async contexts.
///
/// All operations are trivially fulfilled from the in-memory buffer,
/// but the async interface allows uniform code paths.
#[derive(Debug, Clone)]
pub struct AsyncMemoryReader {
    data: Vec<u8>,
}

impl AsyncMemoryReader {
    /// Create a new async memory reader from an owned byte vector.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create a new async memory reader by copying from a byte slice.
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Access the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume and return the underlying bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }
}

impl AsyncHDF5Read for AsyncMemoryReader {
    async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let off = offset as usize;
        if off >= self.data.len() {
            return Ok(Vec::new());
        }
        let end = (off + len).min(self.data.len());
        Ok(self.data[off..end].to_vec())
    }

    async fn len(&self) -> io::Result<u64> {
        Ok(self.data.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// Async dataset reading
// ---------------------------------------------------------------------------

/// Error type for async HDF5 operations.
#[derive(Debug)]
pub enum AsyncHDF5Error {
    /// I/O error during async read.
    Io(io::Error),
    /// HDF5 format parsing error.
    Format(FormatError),
}

impl std::fmt::Display for AsyncHDF5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsyncHDF5Error::Io(e) => write!(f, "I/O error: {e}"),
            AsyncHDF5Error::Format(e) => write!(f, "format error: {e}"),
        }
    }
}

impl std::error::Error for AsyncHDF5Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AsyncHDF5Error::Io(e) => Some(e),
            AsyncHDF5Error::Format(e) => Some(e),
        }
    }
}

impl From<io::Error> for AsyncHDF5Error {
    fn from(e: io::Error) -> Self {
        AsyncHDF5Error::Io(e)
    }
}

impl From<FormatError> for AsyncHDF5Error {
    fn from(e: FormatError) -> Self {
        AsyncHDF5Error::Format(e)
    }
}

/// An async HDF5 file handle that supports reading datasets asynchronously.
///
/// The file data is loaded into memory asynchronously, then parsed using
/// the synchronous rustyhdf5-format parsers (which operate on `&[u8]`).
#[derive(Debug)]
pub struct AsyncHDF5File {
    data: Vec<u8>,
    superblock: Superblock,
}

impl AsyncHDF5File {
    /// Open an HDF5 file asynchronously from any [`AsyncHDF5Read`] source.
    ///
    /// Reads the entire file into memory, then parses the superblock.
    pub async fn open<R: AsyncHDF5Read>(reader: &R) -> Result<Self, AsyncHDF5Error> {
        let data = reader.read_all().await?;
        let sig_offset = find_signature(&data)?;
        let superblock = Superblock::parse(&data, sig_offset)?;
        Ok(Self { data, superblock })
    }

    /// Open an HDF5 file asynchronously from a file path.
    pub async fn open_path<P: AsRef<Path>>(path: P) -> Result<Self, AsyncHDF5Error> {
        let data = tokio::fs::read(path).await?;
        let sig_offset = find_signature(&data)?;
        let superblock = Superblock::parse(&data, sig_offset)?;
        Ok(Self { data, superblock })
    }

    /// Open an HDF5 file from bytes already in memory.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, AsyncHDF5Error> {
        let sig_offset = find_signature(&data)?;
        let superblock = Superblock::parse(&data, sig_offset)?;
        Ok(Self { data, superblock })
    }

    /// Access the raw file bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Access the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Read a dataset at the given path and return raw bytes.
    ///
    /// Resolves the path within the HDF5 group hierarchy, parses the
    /// object header and data layout, then reads the raw data bytes.
    pub async fn read_dataset_raw(&self, path: &str) -> Result<DatasetInfo, AsyncHDF5Error> {
        let addr = resolve_path_any(&self.data, &self.superblock, path)?;
        let header = ObjectHeader::parse(
            &self.data,
            addr as usize,
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;

        let dt_msg = find_msg(&header, MessageType::Datatype)
            .ok_or(FormatError::DatasetMissingData)?;
        let (datatype, _) = Datatype::parse(&dt_msg.data)?;

        let ds_msg = find_msg(&header, MessageType::Dataspace)
            .ok_or(FormatError::DatasetMissingShape)?;
        let dataspace = Dataspace::parse(&ds_msg.data, self.superblock.length_size)?;

        let dl_msg = find_msg(&header, MessageType::DataLayout)
            .ok_or(FormatError::DatasetMissingData)?;
        let layout = DataLayout::parse(
            &dl_msg.data,
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;

        let pipeline = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

        let raw = read_raw_data_full(
            &self.data,
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;

        let shape: Vec<u64> = dataspace.dimensions.clone();

        Ok(DatasetInfo {
            raw,
            datatype,
            shape,
        })
    }

    /// Read a dataset as `f64` values.
    pub async fn read_f64(&self, path: &str) -> Result<Vec<f64>, AsyncHDF5Error> {
        let info = self.read_dataset_raw(path).await?;
        let values = read_as_f64(&info.raw, &info.datatype)?;
        Ok(values)
    }

    /// Read a dataset as `f32` values.
    pub async fn read_f32(&self, path: &str) -> Result<Vec<f32>, AsyncHDF5Error> {
        let info = self.read_dataset_raw(path).await?;
        let values = read_as_f32(&info.raw, &info.datatype)?;
        Ok(values)
    }

    /// Read a dataset as `i32` values.
    pub async fn read_i32(&self, path: &str) -> Result<Vec<i32>, AsyncHDF5Error> {
        let info = self.read_dataset_raw(path).await?;
        let values = read_as_i32(&info.raw, &info.datatype)?;
        Ok(values)
    }

    /// Read a dataset as `i64` values.
    pub async fn read_i64(&self, path: &str) -> Result<Vec<i64>, AsyncHDF5Error> {
        let info = self.read_dataset_raw(path).await?;
        let values = read_as_i64(&info.raw, &info.datatype)?;
        Ok(values)
    }

    /// Read a dataset as `u64` values.
    pub async fn read_u64(&self, path: &str) -> Result<Vec<u64>, AsyncHDF5Error> {
        let info = self.read_dataset_raw(path).await?;
        let values = read_as_u64(&info.raw, &info.datatype)?;
        Ok(values)
    }
}

/// Information about a dataset read from an HDF5 file.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Raw bytes of the dataset data.
    pub raw: Vec<u8>,
    /// The HDF5 datatype of the dataset.
    pub datatype: Datatype,
    /// The shape (dimensions) of the dataset.
    pub shape: Vec<u64>,
}

fn find_msg(header: &ObjectHeader, msg_type: MessageType) -> Option<&HeaderMessage> {
    header.messages.iter().find(|m| m.msg_type == msg_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyhdf5_format::file_writer::FileWriter as FmtWriter;

    fn make_test_hdf5_f64(name: &str, data: &[f64]) -> Vec<u8> {
        let mut fw = FmtWriter::new();
        fw.create_dataset(name).with_f64_data(data);
        fw.finish().unwrap()
    }

    fn make_test_hdf5_i32(name: &str, data: &[i32]) -> Vec<u8> {
        let mut fw = FmtWriter::new();
        fw.create_dataset(name).with_i32_data(data);
        fw.finish().unwrap()
    }

    fn make_test_hdf5_f32(name: &str, data: &[f32]) -> Vec<u8> {
        let mut fw = FmtWriter::new();
        fw.create_dataset(name).with_f32_data(data);
        fw.finish().unwrap()
    }

    // --- AsyncMemoryReader tests ---

    #[tokio::test]
    async fn async_memory_reader_read_at() {
        let reader = AsyncMemoryReader::new(vec![10, 20, 30, 40, 50]);
        let chunk = reader.read_at(1, 3).await.unwrap();
        assert_eq!(chunk, vec![20, 30, 40]);
    }

    #[tokio::test]
    async fn async_memory_reader_len() {
        let reader = AsyncMemoryReader::new(vec![1, 2, 3]);
        assert_eq!(reader.len().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn async_memory_reader_is_empty() {
        let empty = AsyncMemoryReader::new(Vec::new());
        assert!(empty.is_empty().await.unwrap());

        let nonempty = AsyncMemoryReader::new(vec![1]);
        assert!(!nonempty.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn async_memory_reader_read_all() {
        let data = vec![5, 6, 7, 8];
        let reader = AsyncMemoryReader::new(data.clone());
        assert_eq!(reader.read_all().await.unwrap(), data);
    }

    #[tokio::test]
    async fn async_memory_reader_from_slice() {
        let reader = AsyncMemoryReader::from_slice(&[100, 200]);
        assert_eq!(reader.as_bytes(), &[100, 200]);
    }

    #[tokio::test]
    async fn async_memory_reader_into_inner() {
        let reader = AsyncMemoryReader::new(vec![42]);
        assert_eq!(reader.into_inner(), vec![42]);
    }

    #[tokio::test]
    async fn async_memory_reader_read_at_past_end() {
        let reader = AsyncMemoryReader::new(vec![1, 2, 3]);
        let result = reader.read_at(10, 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn async_memory_reader_read_at_partial() {
        let reader = AsyncMemoryReader::new(vec![1, 2, 3]);
        let result = reader.read_at(1, 100).await.unwrap();
        assert_eq!(result, vec![2, 3]);
    }

    // --- AsyncFileReader tests ---

    #[tokio::test]
    async fn async_file_reader_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_async_file_reader_test.bin");

        tokio::fs::write(&path, &[0x89, 0x48, 0x44, 0x46])
            .await
            .unwrap();

        let reader = AsyncFileReader::new(&path);
        let data = reader.read_all().await.unwrap();
        assert_eq!(data, vec![0x89, 0x48, 0x44, 0x46]);
        assert_eq!(reader.len().await.unwrap(), 4);

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn async_file_reader_open_loaded() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_async_file_reader_loaded_test.bin");

        tokio::fs::write(&path, &[1, 2, 3, 4, 5]).await.unwrap();

        let loaded = AsyncFileReader::open(&path).await.unwrap();
        assert_eq!(loaded.as_bytes(), &[1, 2, 3, 4, 5]);

        let chunk = loaded.read_at(2, 2).await.unwrap();
        assert_eq!(chunk, vec![3, 4]);

        assert_eq!(loaded.len().await.unwrap(), 5);

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn async_file_reader_nonexistent() {
        let reader = AsyncFileReader::new("/tmp/rustyhdf5_async_does_not_exist_12345.bin");
        let result = reader.read_all().await;
        assert!(result.is_err());
    }

    // --- AsyncHDF5File tests ---

    #[tokio::test]
    async fn async_hdf5_file_from_bytes() {
        let bytes = make_test_hdf5_f64("data", &[1.0, 2.0, 3.0]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        assert!(!file.as_bytes().is_empty());
        assert!(file.superblock().version <= 3);
    }

    #[tokio::test]
    async fn async_hdf5_file_open_memory_reader() {
        let bytes = make_test_hdf5_f64("data", &[10.0, 20.0]);
        let reader = AsyncMemoryReader::new(bytes);
        let file = AsyncHDF5File::open(&reader).await.unwrap();
        let values = file.read_f64("data").await.unwrap();
        assert_eq!(values, vec![10.0, 20.0]);
    }

    #[tokio::test]
    async fn async_read_f64_dataset() {
        let bytes = make_test_hdf5_f64("values", &[3.14, 2.72, 1.41]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        let values = file.read_f64("values").await.unwrap();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 3.14).abs() < 1e-10);
        assert!((values[1] - 2.72).abs() < 1e-10);
        assert!((values[2] - 1.41).abs() < 1e-10);
    }

    #[tokio::test]
    async fn async_read_i32_dataset() {
        let bytes = make_test_hdf5_i32("ints", &[100, -200, 300]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        let values = file.read_i32("ints").await.unwrap();
        assert_eq!(values, vec![100, -200, 300]);
    }

    #[tokio::test]
    async fn async_read_f32_dataset() {
        let bytes = make_test_hdf5_f32("floats", &[1.5f32, 2.5, 3.5]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        let values = file.read_f32("floats").await.unwrap();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn async_read_dataset_raw_shape() {
        let bytes = make_test_hdf5_f64("matrix", &[1.0, 2.0, 3.0, 4.0]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        let info = file.read_dataset_raw("matrix").await.unwrap();
        assert_eq!(info.shape, vec![4]);
        assert_eq!(info.raw.len(), 4 * 8); // 4 f64 values
    }

    #[tokio::test]
    async fn async_read_dataset_not_found() {
        let bytes = make_test_hdf5_f64("exists", &[1.0]);
        let file = AsyncHDF5File::from_bytes(bytes).unwrap();
        let result = file.read_f64("does_not_exist").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_hdf5_file_open_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_async_hdf5_open_path.h5");

        let bytes = make_test_hdf5_f64("test_data", &[42.0, 84.0]);
        tokio::fs::write(&path, &bytes).await.unwrap();

        let file = AsyncHDF5File::open_path(&path).await.unwrap();
        let values = file.read_f64("test_data").await.unwrap();
        assert_eq!(values, vec![42.0, 84.0]);

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn async_error_display() {
        let io_err = AsyncHDF5Error::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        assert!(io_err.to_string().contains("gone"));

        let fmt_err = AsyncHDF5Error::Format(FormatError::SignatureNotFound);
        assert!(fmt_err.to_string().contains("signature"));
    }

    #[tokio::test]
    async fn async_loaded_reader_into_inner() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_async_loaded_into_inner.bin");
        tokio::fs::write(&path, &[9, 8, 7]).await.unwrap();

        let loaded = AsyncFileReader::open(&path).await.unwrap();
        let inner = loaded.into_inner();
        assert_eq!(inner, vec![9, 8, 7]);

        tokio::fs::remove_file(&path).await.ok();
    }
}
