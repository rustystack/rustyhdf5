#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try common offset/length sizes used in HDF5
    for &offset_size in &[2u8, 4, 8] {
        for &length_size in &[2u8, 4, 8] {
            let _ = purehdf5_format::object_header::ObjectHeader::parse(
                data,
                0,
                offset_size,
                length_size,
            );
        }
    }
});
