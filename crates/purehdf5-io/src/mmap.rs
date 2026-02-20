//! Memory-mapped file readers for zero-copy HDF5 access.
//!
//! Provides [`MmapReader`] for read-only memory-mapped files and
//! [`MmapReadWrite`] for writable memory-mapped files via `memmap2`.

use memmap2::{Mmap, MmapMut};
use std::fs;
use std::io;
use std::path::Path;

use crate::{HDF5Read, HDF5ReadWrite};

/// Memory-mapped file reader for zero-copy access to large files.
///
/// Uses `memmap2` to map the file into the process address space.
/// The key advantage: `read_at()` / `as_bytes()` returns a slice into
/// the mmap — NO COPY.
pub struct MmapReader {
    _file: fs::File,
    mmap: Mmap,
}

impl MmapReader {
    /// Open a file and memory-map it for reading.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying file is not modified
    /// by another process while the mapping is active.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        // SAFETY: We are creating a read-only mapping. The caller is
        // responsible for ensuring the file is not concurrently modified.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Zero-copy access to the entire file contents.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Read a slice at the given offset without copying.
    ///
    /// Returns `None` if `offset + len` exceeds the file size.
    pub fn read_at(&self, offset: usize, len: usize) -> Option<&[u8]> {
        self.mmap.get(offset..offset + len)
    }

    /// Returns the length of the mapped file in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns true if the mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Advise the OS to prefetch the given range (madvise WILLNEED).
    ///
    /// This is a hint to the kernel to start reading the data into memory.
    /// It is safe to call on any platform; on unsupported platforms it is a no-op.
    #[cfg(unix)]
    pub fn advise_willneed(&self, offset: usize, len: usize) {
        let actual_len = len.min(self.mmap.len().saturating_sub(offset));
        if actual_len == 0 {
            return;
        }
        // SAFETY: We are advising on a range within our valid mapping.
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset);
            libc::madvise(ptr as *mut libc::c_void, actual_len, libc::MADV_WILLNEED);
        }
    }

    /// No-op on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn advise_willneed(&self, _offset: usize, _len: usize) {}
}

impl HDF5Read for MmapReader {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Writable memory-mapped file for read-write HDF5 access.
///
/// Uses `memmap2::MmapMut` for mutable memory-mapped files.
pub struct MmapReadWrite {
    _file: fs::File,
    mmap: MmapMut,
}

impl MmapReadWrite {
    /// Open an existing file for read-write memory mapping.
    ///
    /// The file must already exist and have the desired size.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying file is not modified
    /// by another process while the mapping is active.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::OpenOptions::new().read(true).write(true).open(path)?;
        // SAFETY: We create a read-write mapping. Caller ensures no concurrent
        // external modification.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Create a new file of the given size and memory-map it for read-write.
    pub fn create<P: AsRef<Path>>(path: P, size: u64) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(size)?;
        // SAFETY: Fresh file with known size; no concurrent access.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Zero-copy access to the entire file contents.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Mutable access to the entire file contents.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    /// Read a slice at the given offset without copying.
    pub fn read_at(&self, offset: usize, len: usize) -> Option<&[u8]> {
        self.mmap.get(offset..offset + len)
    }

    /// Write data at the given offset.
    ///
    /// Returns an error if the write would exceed the mapped region.
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
        let end = offset + data.len();
        if end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write would exceed mapped region",
            ));
        }
        self.mmap[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    /// Returns the length of the mapped file in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns true if the mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

impl HDF5Read for MmapReadWrite {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

impl HDF5ReadWrite for MmapReadWrite {
    fn write_all_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        if data.len() != self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "data length must match mapped region size for MmapReadWrite",
            ));
        }
        self.mmap.copy_from_slice(data);
        self.mmap.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn mmap_reader_open_and_read() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_read.bin");
        {
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }
        let reader = MmapReader::open(&path).unwrap();
        assert_eq!(reader.as_bytes(), &[1, 2, 3, 4, 5]);
        assert_eq!(reader.len(), 5);
        assert!(!reader.is_empty());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_reader_read_at() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_read_at.bin");
        {
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(&[10, 20, 30, 40, 50]).unwrap();
        }
        let reader = MmapReader::open(&path).unwrap();
        assert_eq!(reader.read_at(1, 3), Some(&[20, 30, 40][..]));
        assert_eq!(reader.read_at(4, 2), None); // out of bounds
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_reader_nonexistent() {
        let result = MmapReader::open("/tmp/purehdf5_mmap_nonexistent_12345.bin");
        assert!(result.is_err());
    }

    #[test]
    fn mmap_readwrite_create_and_write() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_rw.bin");
        {
            let mut rw = MmapReadWrite::create(&path, 5).unwrap();
            rw.write_at(0, &[10, 20, 30, 40, 50]).unwrap();
            rw.flush().unwrap();
        }
        let data = fs::read(&path).unwrap();
        assert_eq!(data, vec![10, 20, 30, 40, 50]);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_readwrite_open_existing() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_rw_open.bin");
        fs::write(&path, [1, 2, 3, 4]).unwrap();
        {
            let mut rw = MmapReadWrite::open(&path).unwrap();
            assert_eq!(rw.as_bytes(), &[1, 2, 3, 4]);
            rw.write_at(2, &[99, 100]).unwrap();
            rw.flush().unwrap();
        }
        let data = fs::read(&path).unwrap();
        assert_eq!(data, vec![1, 2, 99, 100]);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_readwrite_write_all_bytes() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_write_all.bin");
        {
            let mut rw = MmapReadWrite::create(&path, 3).unwrap();
            rw.write_all_bytes(&[7, 8, 9]).unwrap();
        }
        let data = fs::read(&path).unwrap();
        assert_eq!(data, vec![7, 8, 9]);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_readwrite_write_at_out_of_bounds() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_oob.bin");
        let mut rw = MmapReadWrite::create(&path, 3).unwrap();
        let result = rw.write_at(2, &[1, 2, 3]);
        assert!(result.is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_reader_hdf5_read_trait() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_trait.bin");
        fs::write(&path, [0x89, 0x48, 0x44, 0x46]).unwrap();
        let reader = MmapReader::open(&path).unwrap();
        let bytes: &[u8] = HDF5Read::as_bytes(&reader);
        assert_eq!(bytes, &[0x89, 0x48, 0x44, 0x46]);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_readwrite_size_mismatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_mismatch.bin");
        let mut rw = MmapReadWrite::create(&path, 3).unwrap();
        let result = rw.write_all_bytes(&[1, 2, 3, 4, 5]);
        assert!(result.is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_reader_concurrent_references() {
        let dir = std::env::temp_dir();
        let path = dir.join("purehdf5_mmap_test_concurrent.bin");
        fs::write(&path, [1, 2, 3, 4, 5, 6]).unwrap();
        let reader = MmapReader::open(&path).unwrap();

        // Multiple immutable references at the same time
        let slice1 = reader.read_at(0, 3);
        let slice2 = reader.read_at(3, 3);
        let full = reader.as_bytes();

        assert_eq!(slice1, Some(&[1, 2, 3][..]));
        assert_eq!(slice2, Some(&[4, 5, 6][..]));
        assert_eq!(full, &[1, 2, 3, 4, 5, 6]);
        fs::remove_file(&path).ok();
    }
}
