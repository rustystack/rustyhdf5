//! SIMD-accelerated checksum implementations.

/// Compute Fletcher-32 checksum, auto-dispatching to SIMD where available.
pub fn checksum_fletcher32(data: &[u8]) -> u32 {
    match crate::detect_backend() {
        #[cfg(target_arch = "aarch64")]
        crate::Backend::Neon => {
            // SAFETY: aarch64 target verified, NEON always available.
            unsafe { crate::neon::checksum_fletcher32(data) }
        }

        #[cfg(target_arch = "x86_64")]
        crate::Backend::Avx2 | crate::Backend::Avx512 => {
            // SAFETY: Runtime-verified AVX2 support.
            unsafe { crate::avx2::checksum_fletcher32(data) }
        }

        _ => crate::scalar::checksum_fletcher32(data),
    }
}
