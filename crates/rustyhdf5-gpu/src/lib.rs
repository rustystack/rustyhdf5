//! GPU-accelerated vector operations for rustyhdf5.
//!
//! Uses wgpu compute shaders (WGSL) for cross-platform GPU acceleration.
//! Supports CUDA, Metal, Vulkan, and DirectX 12 backends via wgpu.
//!
//! # Usage
//!
//! ```no_run
//! use rustyhdf5_gpu::GpuAccelerator;
//!
//! // Fall back to CPU if GPU is not available
//! let mut gpu = match GpuAccelerator::new() {
//!     Ok(g) => g,
//!     Err(_) => { /* use CPU path */ return; }
//! };
//!
//! // Upload vectors once
//! let vectors = vec![0.0f32; 1000 * 128]; // 1000 vectors of dim 128
//! let norms = vec![1.0f32; 1000];
//! gpu.upload_vectors(&vectors, 128).unwrap();
//! gpu.upload_norms(&norms).unwrap();
//!
//! // Search many times
//! let query = vec![1.0f32; 128];
//! let results = gpu.cosine_search(&query, 10).unwrap();
//! ```

pub mod device;
pub mod error;

#[cfg(feature = "gpu-wgpu")]
mod accelerator;
#[cfg(feature = "gpu-wgpu")]
mod helpers;
#[cfg(feature = "gpu-wgpu")]
mod shaders;

pub use device::DeviceInfo;
pub use error::{GpuError, Result};

#[cfg(feature = "gpu-wgpu")]
pub use accelerator::GpuAccelerator;

/// Stub when compiled without GPU support.
#[cfg(not(feature = "gpu-wgpu"))]
pub struct GpuAccelerator;

#[cfg(not(feature = "gpu-wgpu"))]
impl GpuAccelerator {
    pub fn is_available() -> bool {
        false
    }

    pub fn new() -> Result<Self> {
        Err(GpuError::NotCompiled)
    }
}
