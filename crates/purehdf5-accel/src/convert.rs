//! f16/f32 conversion with SIMD acceleration.

use crate::Backend;

/// Convert a batch of f16 values (as raw u16 bits) to f32.
///
/// Dispatches to the best available SIMD backend.
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    match crate::detect_backend() {
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            crate::neon::f16_to_f32_batch(input, output)
        }

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 | Backend::Avx512 => {
            if is_x86_feature_detected!("f16c") {
                // SAFETY: Runtime-verified f16c support.
                unsafe { crate::avx2::f16_to_f32_batch(input, output) }
            } else {
                crate::scalar::f16_to_f32_batch(input, output)
            }
        }

        _ => crate::scalar::f16_to_f32_batch(input, output),
    }
}
