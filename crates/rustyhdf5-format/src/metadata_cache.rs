//! Metadata cache for parsed object headers and B-tree nodes.
//!
//! The [`MetadataCache`] avoids re-parsing object headers and other metadata
//! structures on repeated accesses. It uses a HashMap keyed by file offset
//! with LRU eviction controlled by a byte budget.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::fmt;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;

/// Default metadata cache size: 2 MiB.
pub const DEFAULT_METADATA_CACHE_BYTES: usize = 2 * 1024 * 1024;

/// Maximum metadata cache size: 32 MiB.
pub const MAX_METADATA_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// A cached metadata entry (opaque bytes representing a parsed structure).
struct CacheEntry {
    /// Cached data (serialized or parsed form).
    data: Vec<u8>,
    /// Approximate size in bytes for budget tracking.
    approx_size: usize,
    /// Monotonic access counter for LRU.
    last_access: u64,
}

/// A metadata cache with LRU eviction.
///
/// Caches parsed object headers and B-tree nodes keyed by file offset.
/// Thread-safe via internal `Mutex`.
pub struct MetadataCache {
    inner: std::sync::Mutex<CacheInner>,
}

struct CacheInner {
    #[cfg(feature = "std")]
    entries: HashMap<u64, CacheEntry>,
    #[cfg(not(feature = "std"))]
    entries: BTreeMap<u64, CacheEntry>,
    current_bytes: usize,
    max_bytes: usize,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl MetadataCache {
    /// Create a new metadata cache with the default size (2 MiB).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_METADATA_CACHE_BYTES)
    }

    /// Create a new metadata cache with a custom byte budget.
    ///
    /// Clamped to [`MAX_METADATA_CACHE_BYTES`].
    pub fn with_capacity(max_bytes: usize) -> Self {
        let max_bytes = max_bytes.min(MAX_METADATA_CACHE_BYTES);
        Self {
            inner: std::sync::Mutex::new(CacheInner {
                #[cfg(feature = "std")]
                entries: HashMap::new(),
                #[cfg(not(feature = "std"))]
                entries: BTreeMap::new(),
                current_bytes: 0,
                max_bytes,
                tick: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Look up cached metadata by file offset.
    ///
    /// Returns a clone of the cached bytes, or `None` if not present.
    pub fn get(&self, offset: u64) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        let result = if let Some(entry) = inner.entries.get_mut(&offset) {
            entry.last_access = tick;
            Some(entry.data.clone())
        } else {
            None
        };
        if result.is_some() {
            inner.hits += 1;
        } else {
            inner.misses += 1;
        }
        result
    }

    /// Insert metadata into the cache.
    ///
    /// If the entry already exists, it is updated. LRU eviction occurs
    /// if the cache exceeds its byte budget.
    pub fn put(&self, offset: u64, data: Vec<u8>) {
        let mut inner = self.inner.lock().unwrap();
        let approx_size = data.len() + 64; // overhead estimate

        // Don't cache if single entry exceeds budget
        if approx_size > inner.max_bytes {
            return;
        }

        inner.tick += 1;
        let tick = inner.tick;

        // Remove existing entry if present
        if let Some(old) = inner.entries.remove(&offset) {
            inner.current_bytes -= old.approx_size;
        }

        // Evict until we have room
        while inner.current_bytes + approx_size > inner.max_bytes && !inner.entries.is_empty() {
            // Find LRU entry
            let lru_key = inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(&k, _)| k)
                .unwrap();
            let removed = inner.entries.remove(&lru_key).unwrap();
            inner.current_bytes -= removed.approx_size;
        }

        inner.current_bytes += approx_size;
        inner.entries.insert(
            offset,
            CacheEntry {
                data,
                approx_size,
                last_access: tick,
            },
        );
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.current_bytes = 0;
        inner.tick = 0;
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().entries.is_empty()
    }

    /// Current bytes used by cached entries.
    pub fn current_bytes(&self) -> usize {
        self.inner.lock().unwrap().current_bytes
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.inner.lock().unwrap().hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.inner.lock().unwrap().misses
    }

    /// Hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        let total = inner.hits + inner.misses;
        if total == 0 {
            0.0
        } else {
            inner.hits as f64 / total as f64
        }
    }
}

impl Default for MetadataCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MetadataCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("MetadataCache")
            .field("entries", &inner.entries.len())
            .field("current_bytes", &inner.current_bytes)
            .field("max_bytes", &inner.max_bytes)
            .field("hits", &inner.hits)
            .field("misses", &inner.misses)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_put_get() {
        let cache = MetadataCache::new();
        cache.put(100, vec![1, 2, 3]);
        assert_eq!(cache.get(100), Some(vec![1, 2, 3]));
        assert_eq!(cache.get(200), None);
    }

    #[test]
    fn lru_eviction() {
        let cache = MetadataCache::with_capacity(256);
        // Each entry ~64 + data_len bytes overhead
        cache.put(1, vec![0; 64]); // ~128 bytes
        cache.put(2, vec![0; 64]); // ~128 bytes = 256 total

        // Access entry 1 to make it more recent
        cache.get(1);

        // Adding entry 3 should evict entry 2 (LRU)
        cache.put(3, vec![0; 64]);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none()); // evicted
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn hit_miss_tracking() {
        let cache = MetadataCache::new();
        cache.put(1, vec![1]);
        cache.get(1); // hit
        cache.get(2); // miss
        cache.get(1); // hit
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn clear_resets() {
        let cache = MetadataCache::new();
        cache.put(1, vec![1, 2, 3]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn oversized_entry_not_cached() {
        let cache = MetadataCache::with_capacity(100);
        cache.put(1, vec![0; 200]); // too big
        assert!(cache.is_empty());
    }
}
