use bytemuck::{Pod, Zeroable};

pub const WORKGROUP_SIZE: u32 = 256;

pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Params4 {
    pub a: u32,
    pub b: u32,
    pub c: f32,
    pub d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Params4U {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

pub fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// CPU-side top-k selection.
/// If `descending` is true, returns the k largest values; otherwise the k smallest.
pub fn top_k_cpu(scores: &[f32], k: usize, descending: bool) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    if descending {
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    indexed.truncate(k);
    indexed
}
