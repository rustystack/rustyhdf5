//! Portable scalar implementations of all operations.
//! These serve as fallbacks when SIMD is not available.

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn vector_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

pub fn batch_cosine(query: &[f32], vectors: &[&[f32]], results: &mut [(usize, f32)]) {
    for (i, v) in vectors.iter().enumerate() {
        results[i] = (i, cosine_similarity(query, v));
    }
}

pub fn batch_cosine_prenorm(
    query_normed: &[f32],
    vectors: &[&[f32]],
    norms: &[f32],
    results: &mut [(usize, f32)],
) {
    for (i, v) in vectors.iter().enumerate() {
        let dot: f32 = query_normed.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
        let sim = if norms[i] == 0.0 { 0.0 } else { dot / norms[i] };
        results[i] = (i, sim);
    }
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

pub fn batch_norms(vectors: &[&[f32]], norms: &mut [f32]) {
    for (i, v) in vectors.iter().enumerate() {
        norms[i] = vector_norm(v);
    }
}

pub fn checksum_fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0xFFFF;
    let mut sum2: u32 = 0xFFFF;

    // Process data as 16-bit words (big-endian, per HDF5 spec)
    let mut i = 0;
    while i + 1 < data.len() {
        let word = ((data[i] as u32) << 8) | (data[i + 1] as u32);
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        i += 2;
    }
    // Handle trailing byte
    if i < data.len() {
        let word = (data[i] as u32) << 8;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }

    (sum2 << 16) | sum1
}

#[cfg(feature = "float16")]
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &bits) in input.iter().enumerate() {
        output[i] = half::f16::from_bits(bits).to_f32();
    }
}

#[cfg(not(feature = "float16"))]
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    // Software f16 -> f32 conversion without external deps
    for (i, &bits) in input.iter().enumerate() {
        output[i] = f16_to_f32_soft(bits);
    }
}

/// Software half-precision to single-precision conversion.
#[cfg(not(feature = "float16"))]
fn f16_to_f32_soft(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Zero
            sign << 31
        } else {
            // Subnormal: normalize
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp32 = (127 - 15 - e) as u32;
            let mant32 = (m & 0x3FF) << 13;
            (sign << 31) | (exp32 << 23) | mant32
        }
    } else if exp == 31 {
        // Inf or NaN
        let mant32 = mant << 13;
        (sign << 31) | (0xFF << 23) | mant32
    } else {
        // Normal
        let exp32 = (exp as i32 - 15 + 127) as u32;
        let mant32 = mant << 13;
        (sign << 31) | (exp32 << 23) | mant32
    };

    f32::from_bits(f32_bits)
}
