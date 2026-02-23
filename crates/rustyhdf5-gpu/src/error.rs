/// Errors for GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no GPU device available")]
    NoDevice,

    #[error("GPU device request failed: {0}")]
    DeviceRequest(String),

    #[error("no vectors uploaded to GPU")]
    NoVectors,

    #[error("no norms uploaded to GPU")]
    NoNorms,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("k ({k}) exceeds vector count ({n})")]
    KExceedsN { k: usize, n: usize },

    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(String),

    #[error("vectors too large for GPU memory: need {need_mb} MB, have {avail_mb} MB")]
    OutOfMemory { need_mb: u64, avail_mb: u64 },

    #[error("GPU feature not enabled — compile with feature 'gpu-wgpu'")]
    NotCompiled,
}

pub type Result<T> = std::result::Result<T, GpuError>;
