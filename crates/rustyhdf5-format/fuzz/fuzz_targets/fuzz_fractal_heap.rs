#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for &offset_size in &[4u8, 8] {
        for &length_size in &[4u8, 8] {
            let _ = rustyhdf5_format::fractal_heap::FractalHeapHeader::parse(
                data,
                0,
                offset_size,
                length_size,
            );
        }
    }
});
