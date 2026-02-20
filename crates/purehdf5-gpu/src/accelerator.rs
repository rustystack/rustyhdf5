use crate::device::DeviceInfo;
use crate::error::{GpuError, Result};
use crate::helpers::{bgl_entry, div_ceil, top_k_cpu, Params4, Params4U, WORKGROUP_SIZE};
use crate::shaders;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// GPU-accelerated vector search engine.
///
/// Upload vectors once, then run many searches against them.
/// If GPU initialization fails, callers should fall back to CPU SIMD.
///
/// Vectors are automatically split into chunks when they exceed the device's
/// `max_storage_buffer_binding_size` (typically 128 MB). Searches dispatch
/// against each chunk and merge results transparently.
pub struct GpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: DeviceInfo,
    max_binding_size: u32,
    vectors_bufs: Vec<wgpu::Buffer>,
    norms_bufs: Vec<wgpu::Buffer>,
    chunk_counts: Vec<usize>,
    dim: usize,
    n_vectors: usize,
}

impl GpuAccelerator {
    /// Check if any GPU is available on this system.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        adapter.is_ok()
    }

    /// Initialize the best available GPU device.
    ///
    /// Requests the adapter's maximum buffer limits so that chunking only
    /// kicks in when truly necessary.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .map_err(|_| GpuError::NoDevice)?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        let info = DeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("purehdf5-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits
                        .max_storage_buffer_binding_size,
                    max_buffer_size: adapter_limits.max_buffer_size,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        ))
        .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            info,
            max_binding_size: adapter_limits.max_storage_buffer_binding_size,
            vectors_bufs: Vec::new(),
            norms_bufs: Vec::new(),
            chunk_counts: Vec::new(),
            dim: 0,
            n_vectors: 0,
        })
    }

    pub fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.max_binding_size
    }

    /// Upload a flat array of vectors to GPU memory.
    /// Automatically splits into chunks when data exceeds the binding limit.
    pub fn upload_vectors(&mut self, vectors: &[f32], dim: usize) -> Result<()> {
        if vectors.is_empty() || dim == 0 {
            return Err(GpuError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let n = vectors.len() / dim;
        if vectors.len() != n * dim {
            return Err(GpuError::DimensionMismatch {
                expected: n * dim,
                got: vectors.len(),
            });
        }

        let max_vecs_per_chunk = self.max_binding_size as usize / (dim * 4);
        if max_vecs_per_chunk == 0 {
            return Err(GpuError::OutOfMemory {
                need_mb: (dim as u64 * 4) / (1024 * 1024),
                avail_mb: self.max_binding_size as u64 / (1024 * 1024),
            });
        }

        let mut bufs = Vec::new();
        let mut counts = Vec::new();
        let mut offset = 0;
        while offset < n {
            let chunk_n = (n - offset).min(max_vecs_per_chunk);
            let start = offset * dim;
            let end = start + chunk_n * dim;
            bufs.push(self.make_storage_buf("vectors_chunk", &vectors[start..end]));
            counts.push(chunk_n);
            offset += chunk_n;
        }

        self.vectors_bufs = bufs;
        self.chunk_counts = counts;
        self.norms_bufs.clear();
        self.dim = dim;
        self.n_vectors = n;
        Ok(())
    }

    /// Upload pre-computed L2 norms, split to match vector chunk layout.
    pub fn upload_norms(&mut self, norms: &[f32]) -> Result<()> {
        if norms.len() != self.n_vectors {
            return Err(GpuError::DimensionMismatch {
                expected: self.n_vectors,
                got: norms.len(),
            });
        }
        let mut bufs = Vec::new();
        let mut offset = 0;
        for &chunk_n in &self.chunk_counts {
            bufs.push(self.make_storage_buf("norms_chunk", &norms[offset..offset + chunk_n]));
            offset += chunk_n;
        }
        self.norms_bufs = bufs;
        Ok(())
    }

    /// Cosine similarity search: returns top-k (index, score) pairs, highest first.
    /// Dispatches against each vector chunk and merges results.
    pub fn cosine_search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        self.check_ready(query.len())?;
        if k > self.n_vectors {
            return Err(GpuError::KExceedsN {
                k,
                n: self.n_vectors,
            });
        }
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dim = self.dim as u32;
        let mut all_results: Vec<(usize, f32)> = Vec::new();
        let mut offset = 0usize;

        for (ci, vecs_buf) in self.vectors_bufs.iter().enumerate() {
            let chunk_n = self.chunk_counts[ci];
            let params = Params4 {
                a: dim,
                b: chunk_n as u32,
                c: query_norm,
                d: 0,
            };
            let scores = self.run_4bind_shader(
                shaders::COSINE_SIMILARITY,
                &params,
                query,
                vecs_buf,
                Some(&self.norms_bufs[ci]),
                chunk_n,
            )?;
            let chunk_topk = top_k_cpu(&scores, k.min(chunk_n), true);
            for (idx, score) in chunk_topk {
                all_results.push((idx + offset, score));
            }
            offset += chunk_n;
        }

        all_results
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    /// Batch cosine search: multiple queries at once.
    pub fn batch_cosine_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        queries.iter().map(|q| self.cosine_search(q, k)).collect()
    }

    /// L2 distance search: returns top-k (index, distance) pairs, smallest first.
    pub fn l2_search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        self.check_vectors(query.len())?;
        if k > self.n_vectors {
            return Err(GpuError::KExceedsN {
                k,
                n: self.n_vectors,
            });
        }
        let dim = self.dim as u32;
        let mut all_results: Vec<(usize, f32)> = Vec::new();
        let mut offset = 0usize;

        for (ci, vecs_buf) in self.vectors_bufs.iter().enumerate() {
            let chunk_n = self.chunk_counts[ci];
            let params = Params4U {
                a: dim,
                b: chunk_n as u32,
                c: 0,
                d: 0,
            };
            let scores =
                self.run_3bind_shader(shaders::L2_DISTANCE, &params, query, vecs_buf, chunk_n)?;
            let chunk_topk = top_k_cpu(&scores, k.min(chunk_n), false);
            for (idx, dist) in chunk_topk {
                all_results.push((idx + offset, dist));
            }
            offset += chunk_n;
        }

        all_results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    /// Compute L2 norms for all uploaded vectors on GPU.
    pub fn compute_norms(&self) -> Result<Vec<f32>> {
        if self.vectors_bufs.is_empty() {
            return Err(GpuError::NoVectors);
        }
        let mut all_norms = Vec::with_capacity(self.n_vectors);
        for (ci, vecs_buf) in self.vectors_bufs.iter().enumerate() {
            let norms = self.run_norms_shader(vecs_buf, self.chunk_counts[ci], self.dim)?;
            all_norms.extend_from_slice(&norms);
        }
        Ok(all_norms)
    }

    /// Compute L2 norms from raw vectors (not previously uploaded).
    /// Handles chunking automatically for large inputs.
    pub fn compute_norms_gpu(&self, vectors: &[f32], dim: usize) -> Result<Vec<f32>> {
        if vectors.is_empty() || dim == 0 {
            return Err(GpuError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let n = vectors.len() / dim;
        if vectors.len() != n * dim {
            return Err(GpuError::DimensionMismatch {
                expected: n * dim,
                got: vectors.len(),
            });
        }
        let max_vecs = self.max_binding_size as usize / (dim * 4);
        let mut all_norms = Vec::with_capacity(n);
        let mut offset = 0;
        while offset < n {
            let chunk_n = (n - offset).min(max_vecs);
            let start = offset * dim;
            let end = start + chunk_n * dim;
            let vecs_buf = self.make_storage_buf("temp_vectors", &vectors[start..end]);
            let norms = self.run_norms_shader(&vecs_buf, chunk_n, dim)?;
            all_norms.extend_from_slice(&norms);
            offset += chunk_n;
        }
        Ok(all_norms)
    }

    /// Batch dot product: queries [Q×D] × vectors [N×D] -> flat [Q×N] scores.
    pub fn batch_dot_product(
        &self,
        queries_flat: &[f32],
        num_queries: usize,
    ) -> Result<Vec<f32>> {
        if self.vectors_bufs.is_empty() {
            return Err(GpuError::NoVectors);
        }
        if queries_flat.len() != num_queries * self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_queries * self.dim,
                got: queries_flat.len(),
            });
        }
        let queries_buf = self.make_storage_buf("queries", queries_flat);
        let mut output = vec![0.0f32; num_queries * self.n_vectors];
        let mut col_offset = 0usize;

        for (ci, vecs_buf) in self.vectors_bufs.iter().enumerate() {
            let chunk_n = self.chunk_counts[ci];
            let total = (num_queries * chunk_n) as u32;
            let params = Params4U {
                a: self.dim as u32,
                b: chunk_n as u32,
                c: num_queries as u32,
                d: 0,
            };
            let chunk_scores = self.run_batch_shader(
                shaders::BATCH_DOT_PRODUCT,
                &params,
                &queries_buf,
                vecs_buf,
                total,
                num_queries * chunk_n,
            )?;
            for qi in 0..num_queries {
                let src = qi * chunk_n;
                let dst = qi * self.n_vectors + col_offset;
                output[dst..dst + chunk_n].copy_from_slice(&chunk_scores[src..src + chunk_n]);
            }
            col_offset += chunk_n;
        }
        Ok(output)
    }

    /// Compute L2 distance matrix: queries × vectors -> Q×N distances.
    /// Uses 16×16 workgroup tiling for cache efficiency.
    pub fn distance_matrix(
        &self,
        queries: &[f32],
        vectors: &[f32],
        dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if queries.is_empty() || vectors.is_empty() || dim == 0 {
            return Err(GpuError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let num_queries = queries.len() / dim;
        let n = vectors.len() / dim;
        if queries.len() != num_queries * dim || vectors.len() != n * dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_queries * dim,
                got: queries.len(),
            });
        }
        let queries_buf = self.make_storage_buf("dm_queries", queries);

        let max_vecs_input = self.max_binding_size as usize / (dim * 4);
        let max_vecs_output = if num_queries > 0 {
            self.max_binding_size as usize / (num_queries * 4)
        } else {
            max_vecs_input
        };
        let max_vecs = max_vecs_input.min(max_vecs_output).max(1);

        let mut flat_output = vec![0.0f32; num_queries * n];
        let mut col_offset = 0usize;
        let mut vec_offset = 0usize;

        while vec_offset < n {
            let chunk_n = (n - vec_offset).min(max_vecs);
            let start = vec_offset * dim;
            let end = start + chunk_n * dim;
            let vecs_buf = self.make_storage_buf("dm_vectors", &vectors[start..end]);
            let params = Params4U {
                a: dim as u32,
                b: chunk_n as u32,
                c: num_queries as u32,
                d: 0,
            };
            let chunk_dists = self.run_distance_matrix_shader(
                &params,
                &queries_buf,
                &vecs_buf,
                num_queries,
                chunk_n,
            )?;
            for qi in 0..num_queries {
                let src = qi * chunk_n;
                let dst = qi * n + col_offset;
                flat_output[dst..dst + chunk_n]
                    .copy_from_slice(&chunk_dists[src..src + chunk_n]);
            }
            col_offset += chunk_n;
            vec_offset += chunk_n;
        }

        Ok((0..num_queries)
            .map(|qi| flat_output[qi * n..(qi + 1) * n].to_vec())
            .collect())
    }

    /// Convert f16 values (as raw u16 bits) to f32 on the GPU.
    pub fn f16_to_f32_batch(&self, f16_bits: &[u16]) -> Result<Vec<f32>> {
        let total = f16_bits.len() as u32;
        let params = Params4U { a: total, b: 0, c: 0, d: 0 };
        let packed: Vec<u32> = f16_bits
            .chunks(2)
            .map(|c| {
                let lo = c[0] as u32;
                let hi = if c.len() > 1 { c[1] as u32 } else { 0 };
                lo | (hi << 16)
            })
            .collect();

        let pair_count = f16_bits.len().div_ceil(2);
        let (_params_buf, _input_buf, output_buf, bgl, bind_group) = self.make_3bind_group(
            &params,
            bytemuck::cast_slice(&packed),
            (f16_bits.len() * 4) as u64,
        );
        let module = self.make_module("f16_to_f32", shaders::F16_TO_F32);
        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(&pipeline, &bind_group, div_ceil(pair_count as u32, WORKGROUP_SIZE));
        self.read_buffer::<f32>(&output_buf, f16_bits.len())
    }

    /// Convert f32 values to f16 (as raw u16 bits) on the GPU.
    pub fn f32_to_f16_batch(&self, values: &[f32]) -> Result<Vec<u16>> {
        let total = values.len() as u32;
        let params = Params4U { a: total, b: 0, c: 0, d: 0 };
        let pair_count = values.len().div_ceil(2);

        let (_params_buf, _input_buf, output_buf, bgl, bind_group) = self.make_3bind_group(
            &params,
            bytemuck::cast_slice(values),
            (pair_count * 4) as u64,
        );
        let module = self.make_module("f32_to_f16", shaders::F32_TO_F16);
        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(&pipeline, &bind_group, div_ceil(pair_count as u32, WORKGROUP_SIZE));

        let packed = self.read_buffer::<u32>(&output_buf, pair_count)?;
        let mut result = Vec::with_capacity(values.len());
        for (i, &word) in packed.iter().enumerate() {
            result.push((word & 0xFFFF) as u16);
            if i * 2 + 1 < values.len() {
                result.push((word >> 16) as u16);
            }
        }
        Ok(result)
    }

    pub fn vector_count(&self) -> usize {
        self.n_vectors
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn chunk_count(&self) -> usize {
        self.chunk_counts.len()
    }

    // ── Internal helpers ──────────────────────────────────────────

    fn check_ready(&self, query_dim: usize) -> Result<()> {
        self.check_vectors(query_dim)?;
        if self.norms_bufs.is_empty() {
            return Err(GpuError::NoNorms);
        }
        Ok(())
    }

    fn check_vectors(&self, query_dim: usize) -> Result<()> {
        if self.vectors_bufs.is_empty() {
            return Err(GpuError::NoVectors);
        }
        if query_dim != self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: self.dim,
                got: query_dim,
            });
        }
        Ok(())
    }

    fn make_storage_buf(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn make_module(&self, label: &str, src: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
    }

    /// Create a 3-binding group: uniform params, storage input, storage RW output.
    #[allow(clippy::type_complexity)]
    fn make_3bind_group(
        &self,
        params: &Params4U,
        input_data: &[u8],
        output_size: u64,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::BindGroupLayout,
        wgpu::BindGroup,
    ) {
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: input_data,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });
        (params_buf, input_buf, output_buf, bgl, bind_group)
    }

    fn run_norms_shader(
        &self,
        vecs_buf: &wgpu::Buffer,
        n: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        let params = Params4U {
            a: dim as u32,
            b: n as u32,
            c: 0,
            d: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norms_out"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = self.make_module("batch_norms", shaders::BATCH_NORMS);
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vecs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });
        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(&pipeline, &bind_group, div_ceil(n as u32, WORKGROUP_SIZE));
        self.read_buffer::<f32>(&output_buf, n)
    }

    fn run_batch_shader(
        &self,
        shader_src: &str,
        params: &Params4U,
        queries_buf: &wgpu::Buffer,
        vecs_buf: &wgpu::Buffer,
        total_threads: u32,
        output_len: usize,
    ) -> Result<Vec<f32>> {
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: (output_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vecs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });
        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(&pipeline, &bind_group, div_ceil(total_threads, WORKGROUP_SIZE));
        self.read_buffer::<f32>(&output_buf, output_len)
    }

    fn run_distance_matrix_shader(
        &self,
        params: &Params4U,
        queries_buf: &wgpu::Buffer,
        vecs_buf: &wgpu::Buffer,
        num_queries: usize,
        chunk_n: usize,
    ) -> Result<Vec<f32>> {
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distances"),
            size: (num_queries * chunk_n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = self.make_module("distance_matrix", shaders::DISTANCE_MATRIX);
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vecs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });
        let pipeline = self.create_pipeline(&module, &bgl);
        let wg_x = div_ceil(chunk_n as u32, 16);
        let wg_y = div_ceil(num_queries as u32, 16);
        self.dispatch_2d(&pipeline, &bind_group, wg_x, wg_y);
        self.read_buffer::<f32>(&output_buf, num_queries * chunk_n)
    }

    fn run_4bind_shader(
        &self,
        shader_src: &str,
        params: &Params4,
        query: &[f32],
        vectors_buf: &wgpu::Buffer,
        extra_buf: Option<&wgpu::Buffer>,
        output_len: usize,
    ) -> Result<Vec<f32>> {
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let query_buf = self.make_storage_buf("query", query);
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: (output_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let mut entries_desc = vec![
            bgl_entry(0, wgpu::BufferBindingType::Uniform),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
        ];
        let mut bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: query_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vectors_buf.as_entire_binding(),
            },
        ];

        if let Some(eb) = extra_buf {
            entries_desc.push(bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }));
            entries_desc.push(bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }));
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: eb.as_entire_binding(),
            });
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            });
        } else {
            entries_desc.push(bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }));
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
            });
        }

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries_desc,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bind_entries,
        });

        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(
            &pipeline,
            &bind_group,
            div_ceil(output_len as u32, WORKGROUP_SIZE),
        );
        self.read_buffer::<f32>(&output_buf, output_len)
    }

    fn run_3bind_shader(
        &self,
        shader_src: &str,
        params: &Params4U,
        query: &[f32],
        vectors_buf: &wgpu::Buffer,
        output_len: usize,
    ) -> Result<Vec<f32>> {
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let query_buf = self.make_storage_buf("query", query);
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: (output_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vectors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });
        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(
            &pipeline,
            &bind_group,
            div_ceil(output_len as u32, WORKGROUP_SIZE),
        );
        self.read_buffer::<f32>(&output_buf, output_len)
    }

    fn create_pipeline(
        &self,
        module: &wgpu::ShaderModule,
        bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[bgl],
                immediate_size: 0,
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch_2d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        wg_x: u32,
        wg_y: u32,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn read_buffer<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<T>> {
        let elem_size = std::mem::size_of::<T>();
        let byte_len = (count * elem_size) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }
}
