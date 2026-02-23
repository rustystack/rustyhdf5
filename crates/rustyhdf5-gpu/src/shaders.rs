//! WGSL compute shaders for GPU-accelerated vector operations.

/// Cosine similarity: each thread computes dot(query, vectors[i]) / norms[i].
/// The query norm is passed as a uniform so we compute full cosine similarity.
pub const COSINE_SIMILARITY: &str = r#"
struct Params {
    dim: u32,
    n: u32,
    query_norm: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read> norms: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n {
        return;
    }
    let dim = params.dim;
    let base = i * dim;
    var dot_val: f32 = 0.0;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        dot_val = dot_val + query[d] * vectors[base + d];
    }
    let denom = params.query_norm * norms[i];
    if denom > 0.0 {
        scores[i] = dot_val / denom;
    } else {
        scores[i] = 0.0;
    }
}
"#;

/// L2 distance: each thread computes ||query - vectors[i]||^2.
pub const L2_DISTANCE: &str = r#"
struct Params {
    dim: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n {
        return;
    }
    let dim = params.dim;
    let base = i * dim;
    var dist: f32 = 0.0;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let diff = query[d] - vectors[base + d];
        dist = dist + diff * diff;
    }
    scores[i] = dist;
}
"#;

/// Batch dot product: queries [Q×D] × vectors [N×D] -> scores [Q×N].
/// Each thread computes one (q, n) pair.
pub const BATCH_DOT_PRODUCT: &str = r#"
struct Params {
    dim: u32,
    n: u32,
    q: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.q * params.n;
    if idx >= total {
        return;
    }
    let qi = idx / params.n;
    let ni = idx % params.n;
    let dim = params.dim;
    let q_base = qi * dim;
    let v_base = ni * dim;
    var dot_val: f32 = 0.0;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        dot_val = dot_val + queries[q_base + d] * vectors[v_base + d];
    }
    scores[qi * params.n + ni] = dot_val;
}
"#;

/// Compute L2 norms for N vectors.
pub const BATCH_NORMS: &str = r#"
struct Params {
    dim: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> norms: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n {
        return;
    }
    let dim = params.dim;
    let base = i * dim;
    var sum: f32 = 0.0;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let v = vectors[base + d];
        sum = sum + v * v;
    }
    norms[i] = sqrt(sum);
}
"#;

/// Convert f16 (stored as u32 pairs) to f32.
/// Input is packed: two f16 values per u32 element.
pub const F16_TO_F32: &str = r#"
struct Params {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn f16_to_f32_manual(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    if exp == 0u {
        if mant == 0u {
            if sign == 1u { return -0.0; } else { return 0.0; }
        }
        // Subnormal
        let f = f32(mant) / 1024.0 * pow(2.0, -14.0);
        if sign == 1u { return -f; } else { return f; }
    }
    if exp == 31u {
        if mant != 0u {
            return bitcast<f32>(0x7FC00000u); // NaN
        }
        if sign == 1u {
            return bitcast<f32>(0xFF800000u); // -Inf
        }
        return bitcast<f32>(0x7F800000u); // +Inf
    }

    let f_exp = f32(i32(exp) - 15);
    let f_mant = 1.0 + f32(mant) / 1024.0;
    let val = f_mant * pow(2.0, f_exp);
    if sign == 1u { return -val; } else { return val; }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let pair_count = (params.total + 1u) / 2u;
    if i >= pair_count {
        return;
    }
    let packed = input[i];
    let lo = packed & 0xFFFFu;
    let hi = (packed >> 16u) & 0xFFFFu;

    let out_idx = i * 2u;
    output[out_idx] = f16_to_f32_manual(lo);
    if out_idx + 1u < params.total {
        output[out_idx + 1u] = f16_to_f32_manual(hi);
    }
}
"#;

/// Convert f32 values to f16 (packed as u32 pairs).
/// Output is packed: two f16 values per u32 element.
pub const F32_TO_F16: &str = r#"
struct Params {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn f32_to_f16_manual(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 31u) & 1u;
    let exp = (bits >> 23u) & 0xFFu;
    let mant = bits & 0x7FFFFFu;

    // Zero
    if exp == 0u && mant == 0u {
        return sign << 15u;
    }
    // Inf or NaN
    if exp == 255u {
        if mant != 0u {
            return (sign << 15u) | 0x7E00u; // NaN
        }
        return (sign << 15u) | 0x7C00u; // Inf
    }
    // f32 subnormal -> f16 zero
    if exp == 0u {
        return sign << 15u;
    }
    // Overflow: exponent > 15 (f32 biased > 142)
    if exp > 142u {
        return (sign << 15u) | 0x7C00u;
    }
    // Underflow to zero: exponent < -24 (f32 biased < 103)
    if exp < 103u {
        return sign << 15u;
    }
    // Subnormal f16: exponent < -14 (f32 biased < 113)
    if exp < 113u {
        let shift = 113u - exp;
        let subnorm = (mant | 0x800000u) >> (shift + 13u);
        return (sign << 15u) | subnorm;
    }
    // Normal f16
    let f16_exp = exp - 112u;
    let f16_mant = mant >> 13u;
    return (sign << 15u) | (f16_exp << 10u) | f16_mant;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let pair_count = (params.total + 1u) / 2u;
    if i >= pair_count {
        return;
    }
    let idx0 = i * 2u;
    let lo = f32_to_f16_manual(input[idx0]);
    var hi: u32 = 0u;
    if idx0 + 1u < params.total {
        hi = f32_to_f16_manual(input[idx0 + 1u]);
    }
    output[i] = lo | (hi << 16u);
}
"#;

/// Distance matrix: computes L2 distance for all (query, vector) pairs.
/// Uses 16×16 workgroup tiling for cache efficiency.
/// Output: Q×N matrix in row-major order.
pub const DISTANCE_MATRIX: &str = r#"
struct Params {
    dim: u32,
    n: u32,
    q: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ni = gid.x;
    let qi = gid.y;
    if qi >= params.q || ni >= params.n {
        return;
    }
    let dim = params.dim;
    let q_base = qi * dim;
    let v_base = ni * dim;
    var dist: f32 = 0.0;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let diff = queries[q_base + d] - vectors[v_base + d];
        dist = dist + diff * diff;
    }
    distances[qi * params.n + ni] = dist;
}
"#;
