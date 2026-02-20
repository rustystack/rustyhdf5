/// Information about a GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {}, max buffer {} MB, max binding {} MB)",
            self.name,
            self.backend,
            self.device_type,
            self.max_buffer_size / (1024 * 1024),
            self.max_storage_buffer_binding_size / (1024 * 1024),
        )
    }
}
