#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Full file parse: signature → superblock → root group object header
    let sig = match rustyhdf5_format::signature::find_signature(data) {
        Ok(s) => s,
        Err(_) => return,
    };
    let sb = match rustyhdf5_format::superblock::Superblock::parse(data, sig) {
        Ok(s) => s,
        Err(_) => return,
    };
    let root_addr = sb.root_group_address as usize;
    let _ = rustyhdf5_format::object_header::ObjectHeader::parse(
        data,
        root_addr,
        sb.offset_size,
        sb.length_size,
    );
    // Also try resolving a path through the root group
    let _ = rustyhdf5_format::group_v2::resolve_path_any(data, &sb, "/");
});
