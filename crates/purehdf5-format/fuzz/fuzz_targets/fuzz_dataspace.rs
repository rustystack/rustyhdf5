#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for &length_size in &[2u8, 4, 8] {
        let _ = purehdf5_format::dataspace::Dataspace::parse(data, length_size);
    }
});
