use crate::device::DeviceInfo;
use crate::error::{GpuError, Result};
use crate::shaders;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params4 {
    a: u32,
    b: u32,
    c: f32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params4U {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// GPU-accelerated vector search engine.
///
/// Upload vectors once, then run many searches against them.
/// If GPU initialization fails, callers should fall back to CPU SIMD.
pub struct GpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: DeviceInfo,
    /// Stored vectors on GPU: N×D f32 values.
    vectors_buf: Option<wgpu::Buffer>,
    /// Pre-computed L2 norms on GPU: N f32 values.
    norms_buf: Option<wgpu::Buffer>,
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
    /// Returns `Err(GpuError::NoDevice)` if no GPU is found.
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
        let limits = adapter.limits();

        let info = DeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            max_buffer_size: limits.max_buffer_size,
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("purehdf5-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            info,
            vectors_buf: None,
            norms_buf: None,
            dim: 0,
            n_vectors: 0,
        })
    }

    /// Returns information about the GPU device.
    pub fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Upload a flat array of vectors to GPU memory.
    ///
    /// `vectors` must have length `n * dim` where `n` is the number of vectors.
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

        let bytes_needed = (vectors.len() * 4) as u64;
        if bytes_needed > self.info.max_buffer_size {
            return Err(GpuError::OutOfMemory {
                need_mb: bytes_needed / (1024 * 1024),
                avail_mb: self.info.max_buffer_size / (1024 * 1024),
            });
        }

        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vectors"),
                contents: bytemuck::cast_slice(vectors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.vectors_buf = Some(buf);
        self.dim = dim;
        self.n_vectors = n;
        Ok(())
    }

    /// Upload pre-computed L2 norms to GPU memory.
    pub fn upload_norms(&mut self, norms: &[f32]) -> Result<()> {
        if norms.len() != self.n_vectors {
            return Err(GpuError::DimensionMismatch {
                expected: self.n_vectors,
                got: norms.len(),
            });
        }
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("norms"),
                contents: bytemuck::cast_slice(norms),
                usage: wgpu::BufferUsages::STORAGE,
            });
        self.norms_buf = Some(buf);
        Ok(())
    }

    /// Cosine similarity search: returns top-k (index, score) pairs, highest first.
    pub fn cosine_search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        self.check_ready(query.len())?;
        let norms_buf = self.norms_buf.as_ref().ok_or(GpuError::NoNorms)?;
        if k > self.n_vectors {
            return Err(GpuError::KExceedsN {
                k,
                n: self.n_vectors,
            });
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n = self.n_vectors as u32;
        let dim = self.dim as u32;

        let params = Params4 {
            a: dim,
            b: n,
            c: query_norm,
            d: 0,
        };

        let scores = self.run_4bind_shader(
            shaders::COSINE_SIMILARITY,
            &params,
            query,
            self.vectors_buf.as_ref().unwrap(),
            Some(norms_buf),
            self.n_vectors,
        )?;

        Ok(top_k_cpu(&scores, k, true))
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

        let n = self.n_vectors as u32;
        let dim = self.dim as u32;

        let params = Params4U {
            a: dim,
            b: n,
            c: 0,
            d: 0,
        };

        let scores = self.run_3bind_shader(
            shaders::L2_DISTANCE,
            &params,
            query,
            self.vectors_buf.as_ref().unwrap(),
            self.n_vectors,
        )?;

        Ok(top_k_cpu(&scores, k, false))
    }

    /// Compute L2 norms for all uploaded vectors on GPU.
    pub fn compute_norms(&self) -> Result<Vec<f32>> {
        let vecs_buf = self.vectors_buf.as_ref().ok_or(GpuError::NoVectors)?;

        let n = self.n_vectors as u32;
        let dim = self.dim as u32;
        let params = Params4U {
            a: dim,
            b: n,
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
            size: (self.n_vectors * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("batch_norms"),
                source: wgpu::ShaderSource::Wgsl(shaders::BATCH_NORMS.into()),
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
                    resource: vecs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(&pipeline, &bind_group, div_ceil(n, WORKGROUP_SIZE));
        self.read_buffer_f32(&output_buf, self.n_vectors)
    }

    /// Batch dot product: queries [Q×D] × vectors [N×D] -> flat [Q×N] scores.
    pub fn batch_dot_product(
        &self,
        queries_flat: &[f32],
        num_queries: usize,
    ) -> Result<Vec<f32>> {
        let vecs_buf = self.vectors_buf.as_ref().ok_or(GpuError::NoVectors)?;
        if queries_flat.len() != num_queries * self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_queries * self.dim,
                got: queries_flat.len(),
            });
        }

        let total = (num_queries * self.n_vectors) as u32;
        let params = Params4U {
            a: self.dim as u32,
            b: self.n_vectors as u32,
            c: num_queries as u32,
            d: 0,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let queries_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("queries"),
                    contents: bytemuck::cast_slice(queries_flat),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: (num_queries * self.n_vectors * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("batch_dot"),
                source: wgpu::ShaderSource::Wgsl(shaders::BATCH_DOT_PRODUCT.into()),
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
        self.dispatch(
            &pipeline,
            &bind_group,
            div_ceil(total, WORKGROUP_SIZE),
        );
        self.read_buffer_f32(&output_buf, num_queries * self.n_vectors)
    }

    /// Convert f16 values (as raw u16 bits) to f32 on the GPU.
    pub fn f16_to_f32_batch(&self, f16_bits: &[u16]) -> Result<Vec<f32>> {
        let total = f16_bits.len() as u32;
        let params = Params4U {
            a: total,
            b: 0,
            c: 0,
            d: 0,
        };

        // Pack pairs of u16 into u32
        let packed: Vec<u32> = f16_bits
            .chunks(2)
            .map(|chunk| {
                let lo = chunk[0] as u32;
                let hi = if chunk.len() > 1 {
                    chunk[1] as u32
                } else {
                    0
                };
                lo | (hi << 16)
            })
            .collect();

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input_f16"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_f32"),
            size: (f16_bits.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("f16_to_f32"),
                source: wgpu::ShaderSource::Wgsl(shaders::F16_TO_F32.into()),
            });

        let pair_count = f16_bits.len().div_ceil(2);
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

        let pipeline = self.create_pipeline(&module, &bgl);
        self.dispatch(
            &pipeline,
            &bind_group,
            div_ceil(pair_count as u32, WORKGROUP_SIZE),
        );
        self.read_buffer_f32(&output_buf, f16_bits.len())
    }

    /// Number of vectors currently uploaded.
    pub fn vector_count(&self) -> usize {
        self.n_vectors
    }

    /// Dimension of uploaded vectors.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    // ── Internal helpers ──────────────────────────────────────────

    fn check_ready(&self, query_dim: usize) -> Result<()> {
        self.check_vectors(query_dim)?;
        if self.norms_buf.is_none() {
            return Err(GpuError::NoNorms);
        }
        Ok(())
    }

    fn check_vectors(&self, query_dim: usize) -> Result<()> {
        if self.vectors_buf.is_none() {
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

    /// Run a shader with layout: uniform, query(storage), vectors(storage), optional(storage), output(storage).
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

        let query_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("query"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
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
            entries_desc.push(bgl_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: true },
            ));
            entries_desc.push(bgl_entry(
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
            ));
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: eb.as_entire_binding(),
            });
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            });
        } else {
            entries_desc.push(bgl_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: false },
            ));
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
        let n = output_len as u32;
        self.dispatch(&pipeline, &bind_group, div_ceil(n, WORKGROUP_SIZE));
        self.read_buffer_f32(&output_buf, output_len)
    }

    /// Run a shader with layout: uniform, query(storage), vectors(storage), output(storage).
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

        let query_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("query"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
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
        let n = output_len as u32;
        self.dispatch(&pipeline, &bind_group, div_ceil(n, WORKGROUP_SIZE));
        self.read_buffer_f32(&output_buf, output_len)
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

    fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
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
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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
fn top_k_cpu(scores: &[f32], k: usize, descending: bool) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    if descending {
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    indexed.truncate(k);
    indexed
}
