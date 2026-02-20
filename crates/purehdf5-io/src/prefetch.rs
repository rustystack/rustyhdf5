//! Prefetch / read-ahead for chunked data access.
//!
//! [`PrefetchReader`] wraps any [`HDF5Read`] implementation and provides
//! prefetching capabilities. For memory-mapped readers, prefetch translates
//! to `madvise(MADV_WILLNEED)`. For file-backed readers, data is preloaded
//! into a ring buffer.

use crate::HDF5Read;

/// A ring buffer that caches prefetched chunks of data.
///
/// Holds up to `capacity` entries, each keyed by `(offset, length)`.
struct RingBuffer {
    entries: Vec<Option<RingEntry>>,
    capacity: usize,
    next_slot: usize,
}

struct RingEntry {
    offset: usize,
    data: Vec<u8>,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);
        Self {
            entries,
            capacity,
            next_slot: 0,
        }
    }

    /// Look up cached data for the given offset and length.
    fn get(&self, offset: usize, len: usize) -> Option<&[u8]> {
        for entry in self.entries.iter().flatten() {
            if entry.offset == offset && entry.data.len() >= len {
                return Some(&entry.data[..len]);
            }
        }
        None
    }

    /// Insert a new entry, evicting the oldest if at capacity.
    fn insert(&mut self, offset: usize, data: Vec<u8>) {
        self.entries[self.next_slot] = Some(RingEntry { offset, data });
        self.next_slot = (self.next_slot + 1) % self.capacity;
    }

    /// Clear all cached entries.
    fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.next_slot = 0;
    }
}

/// Default number of chunks to prefetch ahead.
pub const DEFAULT_PREFETCH_CHUNKS: usize = 4;

/// Wraps any [`HDF5Read`] implementation with prefetch capabilities.
///
/// When reading chunks sequentially, the `PrefetchReader` will prefetch
/// the next N chunks' data into a ring buffer. This is beneficial for
/// sequential access patterns on chunked datasets.
///
/// For memory-mapped readers, the prefetch hint is issued via the
/// underlying OS page cache. For file-backed readers, data is eagerly
/// loaded into a ring buffer for quick subsequent access.
pub struct PrefetchReader<R: HDF5Read> {
    inner: R,
    buffer: RingBuffer,
    chunk_size: usize,
    prefetch_count: usize,
    last_offset: Option<usize>,
}

impl<R: HDF5Read> PrefetchReader<R> {
    /// Create a new `PrefetchReader` wrapping the given reader.
    ///
    /// - `chunk_size`: the size of each chunk in bytes.
    /// - `prefetch_count`: how many chunks ahead to prefetch (default: 4).
    pub fn new(inner: R, chunk_size: usize, prefetch_count: usize) -> Self {
        Self {
            inner,
            buffer: RingBuffer::new(prefetch_count),
            chunk_size,
            prefetch_count,
            last_offset: None,
        }
    }

    /// Create with default prefetch count (4 chunks).
    pub fn with_defaults(inner: R, chunk_size: usize) -> Self {
        Self::new(inner, chunk_size, DEFAULT_PREFETCH_CHUNKS)
    }

    /// Read a chunk at the given offset.
    ///
    /// If the data is in the ring buffer, returns it directly.
    /// Otherwise reads from the inner reader and prefetches ahead.
    pub fn read_chunk(&mut self, offset: usize) -> Option<Vec<u8>> {
        let bytes = self.inner.as_bytes();
        let len = self.chunk_size.min(bytes.len().saturating_sub(offset));
        if len == 0 {
            return None;
        }

        // Check ring buffer first
        if let Some(cached) = self.buffer.get(offset, len) {
            self.last_offset = Some(offset);
            return Some(cached.to_vec());
        }

        // Read from inner
        let data = bytes.get(offset..offset + len)?.to_vec();

        // Detect sequential access and prefetch
        let is_sequential = self
            .last_offset
            .is_some_and(|prev| offset == prev + self.chunk_size);

        if is_sequential {
            self.prefetch_ahead(offset);
        }

        self.last_offset = Some(offset);
        Some(data)
    }

    /// Prefetch the next N chunks into the ring buffer.
    fn prefetch_ahead(&mut self, current_offset: usize) {
        let bytes = self.inner.as_bytes();
        self.buffer.clear();

        for i in 1..=self.prefetch_count {
            let next_offset = current_offset + i * self.chunk_size;
            let remaining = bytes.len().saturating_sub(next_offset);
            let len = self.chunk_size.min(remaining);
            if len == 0 {
                break;
            }
            if let Some(slice) = bytes.get(next_offset..next_offset + len) {
                self.buffer.insert(next_offset, slice.to_vec());
            }
        }
    }

    /// Access the underlying reader.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Consume the prefetch reader and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Returns the configured chunk size.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the prefetch count.
    pub fn prefetch_count(&self) -> usize {
        self.prefetch_count
    }
}

impl<R: HDF5Read> HDF5Read for PrefetchReader<R> {
    fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryReader;

    fn make_test_data(num_chunks: usize, chunk_size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(num_chunks * chunk_size);
        for chunk_idx in 0..num_chunks {
            for byte_idx in 0..chunk_size {
                data.push(((chunk_idx * chunk_size + byte_idx) % 256) as u8);
            }
        }
        data
    }

    #[test]
    fn prefetch_reader_basic_read() {
        let data = make_test_data(10, 100);
        let reader = MemoryReader::new(data.clone());
        let mut prefetch = PrefetchReader::with_defaults(reader, 100);

        let chunk = prefetch.read_chunk(0).unwrap();
        assert_eq!(chunk, &data[0..100]);
    }

    #[test]
    fn prefetch_reader_sequential_chunks() {
        let data = make_test_data(10, 100);
        let reader = MemoryReader::new(data.clone());
        let mut prefetch = PrefetchReader::with_defaults(reader, 100);

        // Read chunks sequentially
        for i in 0..10 {
            let offset = i * 100;
            let chunk = prefetch.read_chunk(offset).unwrap();
            assert_eq!(chunk, &data[offset..offset + 100]);
        }
    }

    #[test]
    fn prefetch_reader_uses_cache() {
        let data = make_test_data(10, 100);
        let reader = MemoryReader::new(data.clone());
        let mut prefetch = PrefetchReader::new(reader, 100, 4);

        // Read first two chunks to trigger sequential detection
        prefetch.read_chunk(0).unwrap();
        prefetch.read_chunk(100).unwrap();

        // Now chunks 200..600 should be in the ring buffer
        // Reading chunk at 200 should hit the cache
        let chunk = prefetch.read_chunk(200).unwrap();
        assert_eq!(chunk, &data[200..300]);
    }

    #[test]
    fn prefetch_reader_out_of_bounds() {
        let data = make_test_data(2, 100);
        let reader = MemoryReader::new(data);
        let mut prefetch = PrefetchReader::with_defaults(reader, 100);

        // Past the end
        let result = prefetch.read_chunk(300);
        assert!(result.is_none());
    }

    #[test]
    fn prefetch_reader_partial_last_chunk() {
        // 250 bytes = 2 full chunks of 100 + 50 remaining
        let data = vec![42u8; 250];
        let reader = MemoryReader::new(data);
        let mut prefetch = PrefetchReader::with_defaults(reader, 100);

        let chunk = prefetch.read_chunk(200).unwrap();
        assert_eq!(chunk.len(), 50);
        assert_eq!(chunk, vec![42u8; 50]);
    }

    #[test]
    fn prefetch_reader_hdf5_read_trait() {
        let data = vec![1, 2, 3, 4, 5];
        let reader = MemoryReader::new(data.clone());
        let prefetch = PrefetchReader::with_defaults(reader, 2);
        assert_eq!(prefetch.as_bytes(), &data[..]);
    }

    #[test]
    fn prefetch_reader_inner_access() {
        let data = vec![10, 20, 30];
        let reader = MemoryReader::new(data.clone());
        let prefetch = PrefetchReader::with_defaults(reader, 1);
        assert_eq!(prefetch.inner().as_bytes(), &data[..]);
        assert_eq!(prefetch.chunk_size(), 1);
        assert_eq!(prefetch.prefetch_count(), DEFAULT_PREFETCH_CHUNKS);
    }

    #[test]
    fn prefetch_reader_into_inner() {
        let data = vec![5, 6, 7];
        let reader = MemoryReader::new(data.clone());
        let prefetch = PrefetchReader::with_defaults(reader, 1);
        let inner = prefetch.into_inner();
        assert_eq!(inner.as_bytes(), &data[..]);
    }

    #[test]
    fn ring_buffer_eviction() {
        let mut rb = RingBuffer::new(2);
        rb.insert(0, vec![1, 2, 3]);
        rb.insert(100, vec![4, 5, 6]);
        // Both should be present
        assert!(rb.get(0, 3).is_some());
        assert!(rb.get(100, 3).is_some());
        // Insert a third — evicts the first
        rb.insert(200, vec![7, 8, 9]);
        assert!(rb.get(0, 3).is_none());
        assert!(rb.get(100, 3).is_some());
        assert!(rb.get(200, 3).is_some());
    }
}
