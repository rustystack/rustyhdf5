# rustyhdf5-netcdf4

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-netcdf4.svg)](https://crates.io/crates/rustyhdf5-netcdf4)
[![docs.rs](https://docs.rs/rustyhdf5-netcdf4/badge.svg)](https://docs.rs/rustyhdf5-netcdf4)

NetCDF-4 read support built on rustyhdf5 — pure Rust, no C dependencies.

## Features

- Read NetCDF-4 / HDF5-backed `.nc` files
- Dimension, variable, and CF convention support
- Climate and scientific data access

## Usage

```rust
use rustyhdf5_netcdf4::NetCDF4File;

let nc = NetCDF4File::open("climate.nc").unwrap();
let temp = nc.variable("temperature").unwrap();
```

## License

MIT
