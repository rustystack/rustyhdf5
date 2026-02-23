#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try parsing with signature at offset 0
    let _ = rustyhdf5_format::superblock::Superblock::parse(data, 0);
    // Try with a valid-ish signature search first
    if let Ok(offset) = rustyhdf5_format::signature::find_signature(data) {
        let _ = rustyhdf5_format::superblock::Superblock::parse(data, offset);
    }
});
