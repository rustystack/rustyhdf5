//! Chunk cache with hash-based index and LRU eviction.
//!
//! The [`ChunkCache`] avoids re-traversing B-trees on repeated reads of chunked
//! datasets.  On first access it scans the B-tree once and builds a
//! `HashMap<ChunkCoord, ChunkInfo>` (the *chunk index*).  Decompressed chunk
//! data is cached with LRU eviction controlled by a byte-budget.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::cell::RefCell;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;

use crate::chunked_read::ChunkInfo;

/// Coordinate key for a chunk — the N-dimensional offset vector.
pub type ChunkCoord = Vec<u64>;

/// Default maximum bytes of decompressed chunk data to cache.
pub const DEFAULT_CACHE_BYTES: usize = 1024 * 1024; // 1 MiB

/// Default maximum number of cached decompressed chunks.
pub const DEFAULT_MAX_SLOTS: usize = 16;

// ---------------------------------------------------------------------------
// LRU entry
// ---------------------------------------------------------------------------

struct CachedChunk {
    coord: ChunkCoord,
    data: Vec<u8>,
    /// Monotonically increasing access counter for LRU ordering.
    last_access: u64,
}

// ---------------------------------------------------------------------------
// ChunkCache
// ---------------------------------------------------------------------------

/// A per-dataset chunk cache with hash-based index and LRU eviction.
///
/// # Usage
///
/// ```ignore
/// let cache = ChunkCache::new();
/// // Pass &cache to read_chunked_data — it will populate the index lazily.
/// ```
///
/// The cache is wrapped in `RefCell` internally so it can be mutated through
/// shared references (single-threaded use).
pub struct ChunkCache {
    inner: RefCell<CacheInner>,
}

struct CacheInner {
    /// Hash index: chunk coordinate → ChunkInfo (offset + size in file).
    /// Populated once per dataset on first access.
    #[cfg(feature = "std")]
    index: Option<HashMap<ChunkCoord, ChunkInfo>>,
    #[cfg(not(feature = "std"))]
    index: Option<BTreeMap<ChunkCoord, ChunkInfo>>,

    /// LRU cache of decompressed chunk data.
    slots: Vec<CachedChunk>,

    /// Current total bytes of cached decompressed data.
    current_bytes: usize,

    /// Maximum bytes of decompressed data to cache.
    max_bytes: usize,

    /// Maximum number of slots.
    max_slots: usize,

    /// Monotonic counter for LRU ordering.
    tick: u64,
}

impl ChunkCache {
    /// Create a new chunk cache with default limits (1 MiB, 16 slots).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_BYTES, DEFAULT_MAX_SLOTS)
    }

    /// Create a new chunk cache with custom byte budget and slot count.
    pub fn with_capacity(max_bytes: usize, max_slots: usize) -> Self {
        Self {
            inner: RefCell::new(CacheInner {
                index: None,
                slots: Vec::with_capacity(max_slots.min(64)),
                current_bytes: 0,
                max_bytes,
                max_slots,
                tick: 0,
            }),
        }
    }

    // ----- Index operations -----

    /// Returns `true` if the chunk index has been built.
    pub fn has_index(&self) -> bool {
        self.inner.borrow().index.is_some()
    }

    /// Build the chunk index from a pre-collected list of `ChunkInfo`.
    ///
    /// The `rank` parameter is used to truncate offsets to spatial dims only
    /// (B-tree v1 stores rank+1 offsets).
    pub fn populate_index(&self, chunks: &[ChunkInfo], rank: usize) {
        let mut inner = self.inner.borrow_mut();
        if inner.index.is_some() {
            return; // already populated
        }
        #[cfg(feature = "std")]
        let mut map = HashMap::with_capacity(chunks.len());
        #[cfg(not(feature = "std"))]
        let mut map = BTreeMap::new();

        for ci in chunks {
            let coord: ChunkCoord = ci.offsets.iter().take(rank).copied().collect();
            map.insert(coord, ci.clone());
        }
        inner.index = Some(map);
    }

    /// Look up a chunk by its spatial coordinate in the index.
    pub fn lookup_index(&self, coord: &[u64]) -> Option<ChunkInfo> {
        let inner = self.inner.borrow();
        inner.index.as_ref()?.get(coord).cloned()
    }

    /// Return all indexed chunks as a `Vec<ChunkInfo>` (order unspecified).
    pub fn all_indexed_chunks(&self) -> Option<Vec<ChunkInfo>> {
        let inner = self.inner.borrow();
        inner.index.as_ref().map(|m| m.values().cloned().collect())
    }

    // ----- Decompressed data cache (LRU) -----

    /// Try to get cached decompressed data for a chunk coordinate.
    pub fn get_decompressed(&self, coord: &[u64]) -> Option<Vec<u8>> {
        let mut inner = self.inner.borrow_mut();
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord.as_slice() == coord {
                slot.last_access = tick;
                return Some(slot.data.clone());
            }
        }
        None
    }

    /// Insert decompressed chunk data into the LRU cache.
    pub fn put_decompressed(&self, coord: ChunkCoord, data: Vec<u8>) {
        let mut inner = self.inner.borrow_mut();
        let data_len = data.len();

        // Don't cache if single chunk exceeds budget
        if data_len > inner.max_bytes {
            return;
        }

        // Check if already present
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord == coord {
                slot.last_access = tick;
                return; // already cached
            }
        }

        // Evict until we have room
        while inner.slots.len() >= inner.max_slots
            || (inner.current_bytes + data_len > inner.max_bytes && !inner.slots.is_empty())
        {
            // Find LRU slot
            let lru_idx = inner
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_access)
                .map(|(i, _)| i)
                .unwrap();
            let removed = inner.slots.swap_remove(lru_idx);
            inner.current_bytes -= removed.data.len();
        }

        inner.current_bytes += data_len;
        inner.slots.push(CachedChunk {
            coord,
            data,
            last_access: tick,
        });
    }

    /// Clear the entire cache (index + decompressed data).
    pub fn clear(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.index = None;
        inner.slots.clear();
        inner.current_bytes = 0;
        inner.tick = 0;
    }

    /// Number of decompressed chunks currently cached.
    pub fn cached_chunk_count(&self) -> usize {
        self.inner.borrow().slots.len()
    }

    /// Total bytes of decompressed data currently cached.
    pub fn cached_bytes(&self) -> usize {
        self.inner.borrow().current_bytes
    }
}

impl Default for ChunkCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(offsets: Vec<u64>, address: u64, size: u32) -> ChunkInfo {
        ChunkInfo {
            chunk_size: size,
            filter_mask: 0,
            offsets,
            address,
        }
    }

    #[test]
    fn index_populate_and_lookup() {
        let cache = ChunkCache::new();
        let chunks = vec![
            make_chunk(vec![0, 0, 0], 0x1000, 80),
            make_chunk(vec![10, 0, 0], 0x2000, 80),
        ];
        cache.populate_index(&chunks, 2); // rank=2, truncate to [0,0] and [10,0]
        assert!(cache.has_index());

        let c0 = cache.lookup_index(&[0, 0]).unwrap();
        assert_eq!(c0.address, 0x1000);

        let c1 = cache.lookup_index(&[10, 0]).unwrap();
        assert_eq!(c1.address, 0x2000);

        assert!(cache.lookup_index(&[5, 0]).is_none());
    }

    #[test]
    fn decompressed_cache_hit() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0, 0], vec![1, 2, 3, 4]);
        let got = cache.get_decompressed(&[0, 0]).unwrap();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }

    #[test]
    fn lru_eviction_by_slots() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2); // max 2 slots

        cache.put_decompressed(vec![0], vec![1; 10]);
        cache.put_decompressed(vec![1], vec![2; 10]);
        assert_eq!(cache.cached_chunk_count(), 2);

        // Access slot 0 to make it more recent
        cache.get_decompressed(&[0]);

        // Insert slot 2 — should evict slot 1 (LRU)
        cache.put_decompressed(vec![2], vec![3; 10]);
        assert_eq!(cache.cached_chunk_count(), 2);

        assert!(cache.get_decompressed(&[0]).is_some());
        assert!(cache.get_decompressed(&[1]).is_none()); // evicted
        assert!(cache.get_decompressed(&[2]).is_some());
    }

    #[test]
    fn lru_eviction_by_bytes() {
        let cache = ChunkCache::with_capacity(50, 100); // 50 bytes max

        cache.put_decompressed(vec![0], vec![0; 20]);
        cache.put_decompressed(vec![1], vec![0; 20]);
        assert_eq!(cache.cached_bytes(), 40);

        // This needs 20 bytes but only 10 free — evict LRU
        cache.put_decompressed(vec![2], vec![0; 20]);
        assert!(cache.cached_bytes() <= 50);
        assert!(cache.get_decompressed(&[0]).is_none()); // evicted (LRU)
    }

    #[test]
    fn oversized_chunk_not_cached() {
        let cache = ChunkCache::with_capacity(10, 16);
        cache.put_decompressed(vec![0], vec![0; 100]); // too big
        assert_eq!(cache.cached_chunk_count(), 0);
    }

    #[test]
    fn clear_resets_everything() {
        let cache = ChunkCache::new();
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        cache.put_decompressed(vec![0], vec![1, 2, 3]);

        cache.clear();
        assert!(!cache.has_index());
        assert_eq!(cache.cached_chunk_count(), 0);
        assert_eq!(cache.cached_bytes(), 0);
    }

    #[test]
    fn duplicate_insert_is_noop() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0], vec![1, 2, 3]);
        cache.put_decompressed(vec![0], vec![1, 2, 3]); // duplicate
        assert_eq!(cache.cached_chunk_count(), 1);
        assert_eq!(cache.cached_bytes(), 3);
    }
}
