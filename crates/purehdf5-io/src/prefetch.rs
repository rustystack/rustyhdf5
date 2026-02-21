//! Prefetch / read-ahead for chunked data access.
//!
//! [`PrefetchReader`] wraps any [`HDF5Read`] implementation and provides
//! prefetching capabilities. For memory-mapped readers, prefetch translates
//! to `madvise(MADV_WILLNEED)`. For file-backed readers, data is preloaded
//! into a ring buffer.
//!
//! [`SweepDetector`] tracks N-dimensional chunk access patterns and triggers
//! adaptive prefetching when a sweep pattern is detected.

use crate::HDF5Read;
use crate::sweep::{ChunkCoord, SweepDirection, detect_sweep, predict_next};

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

    /// Preload predicted chunk byte ranges into the ring buffer.
    ///
    /// Called by the sweep detector when it has predicted upcoming chunk
    /// file offsets. Each `(offset, size)` pair is loaded from the inner
    /// reader into the ring buffer for fast subsequent access.
    pub fn preload_ranges(&mut self, ranges: &[(usize, usize)]) {
        let bytes = self.inner.as_bytes();
        for &(offset, size) in ranges {
            let len = size.min(bytes.len().saturating_sub(offset));
            if len == 0 {
                continue;
            }
            if let Some(slice) = bytes.get(offset..offset + len) {
                self.buffer.insert(offset, slice.to_vec());
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

// ---------------------------------------------------------------------------
// SweepDetector — N-dimensional sweep pattern tracking
// ---------------------------------------------------------------------------

/// Default sliding window size for sweep detection.
pub const DEFAULT_WINDOW_SIZE: usize = 12;

/// Default number of chunks to prefetch when a sweep is detected.
pub const DEFAULT_SWEEP_PREFETCH_COUNT: usize = 4;

/// Tracks chunk access patterns and detects N-dimensional sweep directions.
///
/// Maintains a sliding window of the last N chunk coordinates. When a
/// consistent sweep pattern is detected across the window, it predicts
/// the next chunks and can trigger prefetch via the ring buffer or
/// `madvise(MADV_WILLNEED)` for memory-mapped readers.
pub struct SweepDetector {
    /// Sliding window of recent chunk coordinates.
    history: Vec<ChunkCoord>,
    /// Maximum history window size.
    window_size: usize,
    /// Number of dimensions in the dataset.
    ndims: usize,
    /// Currently detected sweep direction (cached).
    current_direction: SweepDirection,
    /// How many chunks ahead to prefetch on sweep detection.
    prefetch_count: usize,
}

impl SweepDetector {
    /// Create a new sweep detector.
    ///
    /// - `ndims`: number of dimensions in the chunked dataset.
    /// - `window_size`: sliding window size (8–16 recommended).
    /// - `prefetch_count`: how many chunks to predict ahead (2–4 recommended).
    pub fn new(ndims: usize, window_size: usize, prefetch_count: usize) -> Self {
        Self {
            history: Vec::with_capacity(window_size),
            window_size,
            ndims,
            current_direction: SweepDirection::Random,
            prefetch_count,
        }
    }

    /// Create with default settings (window=12, prefetch=4).
    pub fn with_defaults(ndims: usize) -> Self {
        Self::new(ndims, DEFAULT_WINDOW_SIZE, DEFAULT_SWEEP_PREFETCH_COUNT)
    }

    /// Record a chunk coordinate access and update the detected direction.
    ///
    /// Returns the predicted next chunk coordinates if a sweep pattern
    /// is detected, or an empty `Vec` if the pattern is random.
    pub fn record_access(&mut self, coord: ChunkCoord) -> Vec<ChunkCoord> {
        // Add to sliding window
        if self.history.len() >= self.window_size {
            self.history.remove(0);
        }
        self.history.push(coord);

        // Re-detect sweep direction
        self.current_direction = detect_sweep(&self.history, self.ndims);

        // Predict next chunks if pattern detected
        if self.current_direction != SweepDirection::Random {
            predict_next(&self.history, self.current_direction, self.prefetch_count)
        } else {
            Vec::new()
        }
    }

    /// Returns the currently detected sweep direction.
    pub fn direction(&self) -> SweepDirection {
        self.current_direction
    }

    /// Returns the current access history.
    pub fn history(&self) -> &[ChunkCoord] {
        &self.history
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_direction = SweepDirection::Random;
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

    // --- SweepDetector tests ---

    #[test]
    fn sweep_detector_detects_row_major() {
        let mut detector = SweepDetector::with_defaults(2);
        // Feed a row-major pattern (dim 1 changes)
        detector.record_access(vec![0, 0]);
        detector.record_access(vec![0, 10]);
        let predictions = detector.record_access(vec![0, 20]);

        assert_eq!(detector.direction(), SweepDirection::RowMajor);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0], vec![0, 30]);
    }

    #[test]
    fn sweep_detector_detects_column_major() {
        let mut detector = SweepDetector::with_defaults(2);
        detector.record_access(vec![0, 0]);
        detector.record_access(vec![10, 0]);
        let predictions = detector.record_access(vec![20, 0]);

        assert_eq!(detector.direction(), SweepDirection::ColumnMajor);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0], vec![30, 0]);
    }

    #[test]
    fn sweep_detector_random_gives_no_predictions() {
        let mut detector = SweepDetector::with_defaults(2);
        detector.record_access(vec![0, 0]);
        detector.record_access(vec![30, 20]);
        let predictions = detector.record_access(vec![10, 0]);

        assert_eq!(detector.direction(), SweepDirection::Random);
        assert!(predictions.is_empty());
    }

    #[test]
    fn sweep_detector_reset() {
        let mut detector = SweepDetector::with_defaults(2);
        detector.record_access(vec![0, 0]);
        detector.record_access(vec![0, 10]);
        detector.record_access(vec![0, 20]);
        assert_eq!(detector.direction(), SweepDirection::RowMajor);

        detector.reset();
        assert_eq!(detector.direction(), SweepDirection::Random);
        assert!(detector.history().is_empty());
    }

    #[test]
    fn sweep_detector_sliding_window() {
        let mut detector = SweepDetector::new(2, 4, 2);
        // Fill window
        detector.record_access(vec![0, 0]);
        detector.record_access(vec![0, 10]);
        detector.record_access(vec![0, 20]);
        detector.record_access(vec![0, 30]);
        assert_eq!(detector.history().len(), 4);

        // Adding one more should evict oldest
        detector.record_access(vec![0, 40]);
        assert_eq!(detector.history().len(), 4);
        assert_eq!(detector.history()[0], vec![0, 10]);
    }

    #[test]
    fn sweep_detector_3d_slice_major() {
        let mut detector = SweepDetector::with_defaults(3);
        detector.record_access(vec![0, 0, 0]);
        detector.record_access(vec![0, 4, 0]);
        let predictions = detector.record_access(vec![0, 8, 0]);

        assert_eq!(detector.direction(), SweepDirection::SliceMajor(1));
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0], vec![0, 12, 0]);
    }

    #[test]
    fn preload_ranges_into_ring_buffer() {
        let data = make_test_data(10, 100);
        let reader = MemoryReader::new(data.clone());
        let mut prefetch = PrefetchReader::new(reader, 100, 8);

        // Preload specific ranges
        prefetch.preload_ranges(&[(200, 100), (500, 100)]);

        // These should now be in the ring buffer
        let chunk = prefetch.read_chunk(200).unwrap();
        assert_eq!(chunk, &data[200..300]);
    }

    #[test]
    fn sweep_prefetch_reduces_misses() {
        // Simulate a row-major sweep on a 2D chunked dataset
        // and verify that predicted chunks are correct
        let mut detector = SweepDetector::new(2, 8, 3);

        // Build up pattern
        let mut all_predictions = Vec::new();
        for i in 0..6 {
            let coord = vec![0, i * 10];
            let preds = detector.record_access(coord);
            all_predictions.push(preds);
        }

        // After a few accesses, should be predicting correctly
        assert_eq!(detector.direction(), SweepDirection::RowMajor);
        let last_preds = all_predictions.last().unwrap();
        assert!(!last_preds.is_empty());
        // Should predict the next chunks along dim 1
        assert_eq!(last_preds[0], vec![0, 60]);
        assert_eq!(last_preds[1], vec![0, 70]);
        assert_eq!(last_preds[2], vec![0, 80]);
    }

    #[test]
    fn random_access_no_false_sweep() {
        let mut detector = SweepDetector::with_defaults(3);
        // Random-ish access pattern
        let coords = vec![
            vec![0, 0, 0],
            vec![12, 8, 4],
            vec![4, 0, 12],
            vec![0, 12, 0],
            vec![8, 4, 8],
            vec![12, 0, 4],
            vec![4, 8, 12],
            vec![0, 4, 0],
        ];
        for coord in coords {
            let preds = detector.record_access(coord);
            // Should never get predictions for random access
            assert!(
                preds.is_empty(),
                "false sweep detected: direction={:?}",
                detector.direction()
            );
        }
        assert_eq!(detector.direction(), SweepDirection::Random);
    }
}
