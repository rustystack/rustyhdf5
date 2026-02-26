//! I/O profiling hooks for monitoring read and cache performance.
//!
//! The [`IoProfiler`] trait defines callbacks for read, decompress, and cache
//! events. The [`DefaultProfiler`] implementation uses atomic counters for
//! thread-safe, low-overhead profiling.

use core::sync::atomic::{AtomicU64, Ordering};

/// Trait for profiling I/O operations.
///
/// Implement this trait to collect custom metrics or forward events to
/// an external monitoring system.
pub trait IoProfiler: Send + Sync {
    /// Called when raw bytes are read from the file.
    fn on_read(&self, bytes: u64);

    /// Called when a chunk is decompressed.
    fn on_decompress(&self, compressed_bytes: u64, decompressed_bytes: u64);

    /// Called when a chunk cache hit occurs.
    fn on_cache_hit(&self, bytes: u64);

    /// Called when a chunk cache miss occurs.
    fn on_cache_miss(&self);
}

/// Default profiler using atomic counters.
///
/// Thread-safe and allocation-free after construction. Suitable for
/// production use with minimal overhead.
#[derive(Debug)]
pub struct DefaultProfiler {
    /// Total bytes read from file.
    pub bytes_read: AtomicU64,
    /// Total number of read operations.
    pub read_count: AtomicU64,
    /// Total bytes decompressed.
    pub bytes_decompressed: AtomicU64,
    /// Total compressed bytes input to decompression.
    pub bytes_compressed_in: AtomicU64,
    /// Number of decompress operations.
    pub decompress_count: AtomicU64,
    /// Number of cache hits.
    pub cache_hits: AtomicU64,
    /// Bytes served from cache.
    pub cache_hit_bytes: AtomicU64,
    /// Number of cache misses.
    pub cache_misses: AtomicU64,
}

impl DefaultProfiler {
    /// Create a new profiler with all counters at zero.
    pub fn new() -> Self {
        Self {
            bytes_read: AtomicU64::new(0),
            read_count: AtomicU64::new(0),
            bytes_decompressed: AtomicU64::new(0),
            bytes_compressed_in: AtomicU64::new(0),
            decompress_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_hit_bytes: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Total bytes read from file.
    pub fn total_bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Total read operations.
    pub fn total_reads(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }

    /// Total bytes decompressed.
    pub fn total_bytes_decompressed(&self) -> u64 {
        self.bytes_decompressed.load(Ordering::Relaxed)
    }

    /// Compression ratio (decompressed / compressed). Returns 0.0 if no
    /// decompression has occurred.
    pub fn compression_ratio(&self) -> f64 {
        let compressed = self.bytes_compressed_in.load(Ordering::Relaxed);
        let decompressed = self.bytes_decompressed.load(Ordering::Relaxed);
        if compressed == 0 {
            0.0
        } else {
            decompressed as f64 / compressed as f64
        }
    }

    /// Cache hit rate as a fraction in [0.0, 1.0].
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
        self.read_count.store(0, Ordering::Relaxed);
        self.bytes_decompressed.store(0, Ordering::Relaxed);
        self.bytes_compressed_in.store(0, Ordering::Relaxed);
        self.decompress_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_hit_bytes.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
}

impl Default for DefaultProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl IoProfiler for DefaultProfiler {
    fn on_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.read_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_decompress(&self, compressed_bytes: u64, decompressed_bytes: u64) {
        self.bytes_compressed_in
            .fetch_add(compressed_bytes, Ordering::Relaxed);
        self.bytes_decompressed
            .fetch_add(decompressed_bytes, Ordering::Relaxed);
        self.decompress_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_cache_hit(&self, bytes: u64) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.cache_hit_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn on_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profiler_basic() {
        let p = DefaultProfiler::new();
        p.on_read(1024);
        p.on_read(2048);
        assert_eq!(p.total_bytes_read(), 3072);
        assert_eq!(p.total_reads(), 2);
    }

    #[test]
    fn default_profiler_decompress() {
        let p = DefaultProfiler::new();
        p.on_decompress(100, 1000);
        p.on_decompress(200, 2000);
        assert_eq!(p.total_bytes_decompressed(), 3000);
        assert!((p.compression_ratio() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn default_profiler_cache() {
        let p = DefaultProfiler::new();
        p.on_cache_hit(512);
        p.on_cache_hit(512);
        p.on_cache_miss();
        assert!((p.cache_hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn default_profiler_reset() {
        let p = DefaultProfiler::new();
        p.on_read(100);
        p.on_cache_hit(50);
        p.reset();
        assert_eq!(p.total_bytes_read(), 0);
        assert_eq!(p.cache_hit_rate(), 0.0);
    }
}
