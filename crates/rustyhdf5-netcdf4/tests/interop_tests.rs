//! NetCDF-4 interop tests: Python creates NetCDF-4 files, rustyhdf5-netcdf4 reads them.
//!
//! Tests are skipped if python3 or netCDF4/xarray Python packages are not available.

use std::process::Command;

use rustyhdf5_netcdf4::{AttrValue, NetCDF4File};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn netcdf4_python_available() -> bool {
    Command::new("python3")
        .args(["-c", "import netCDF4; print(netCDF4.__version__)"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn xarray_available() -> bool {
    Command::new("python3")
        .args(["-c", "import xarray; print(xarray.__version__)"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

macro_rules! skip_if_no_netcdf4 {
    () => {
        if !netcdf4_python_available() {
            eprintln!("SKIP: python3 with netCDF4 not available");
            return;
        }
    };
}

macro_rules! skip_if_no_xarray {
    () => {
        if !xarray_available() {
            eprintln!("SKIP: python3 with xarray not available");
            return;
        }
    };
}

fn run_python(script: &str) {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("failed to run python3");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Python script failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        );
    }
}

// ===========================================================================
// 1. Python netCDF4 creates file with dims, vars, CF attrs -> read with rustyhdf5
// ===========================================================================

#[test]
fn netcdf4_python_creates_cf_file_rustyhdf5_reads() {
    skip_if_no_netcdf4!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cf_test.nc");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("{path_str}", "w", format="NETCDF4")
ds.Conventions = "CF-1.8"
ds.title = "Test CF Dataset"

# Dimensions
lat_dim = ds.createDimension("lat", 3)
lon_dim = ds.createDimension("lon", 4)
time_dim = ds.createDimension("time", None)  # unlimited

# Coordinate variables
lat = ds.createVariable("lat", "f4", ("lat",))
lat.units = "degrees_north"
lat.standard_name = "latitude"
lat[:] = [10.0, 20.0, 30.0]

lon = ds.createVariable("lon", "f4", ("lon",))
lon.units = "degrees_east"
lon.standard_name = "longitude"
lon[:] = [-120.0, -110.0, -100.0, -90.0]

time = ds.createVariable("time", "f8", ("time",))
time.units = "hours since 2000-01-01"
time.calendar = "standard"
time[:] = [0.0, 6.0, 12.0]

# Data variable
temp = ds.createVariable("temperature", "f4", ("time", "lat", "lon"),
                         fill_value=-9999.0)
temp.units = "K"
temp.long_name = "Air Temperature"
temp.standard_name = "air_temperature"
data = np.arange(36, dtype=np.float32).reshape(3, 3, 4) + 270.0
temp[:] = data

ds.close()
"#
    );
    run_python(&script);

    let file = NetCDF4File::open(&path).unwrap();

    // Check dimensions
    let dims = file.dimensions().unwrap();
    let dim_names: Vec<&str> = dims.iter().map(|d| d.name.as_str()).collect();
    assert!(dim_names.contains(&"lat"));
    assert!(dim_names.contains(&"lon"));
    assert!(dim_names.contains(&"time"));

    let lat_dim = dims.iter().find(|d| d.name == "lat").unwrap();
    assert_eq!(lat_dim.size, 3);

    let lon_dim = dims.iter().find(|d| d.name == "lon").unwrap();
    assert_eq!(lon_dim.size, 4);

    let time_dim = dims.iter().find(|d| d.name == "time").unwrap();
    assert_eq!(time_dim.size, 3);

    // Check variables
    let variables = file.variables().unwrap();
    let var_names: Vec<String> = variables.iter().map(|v| v.name().to_string()).collect();
    assert!(var_names.contains(&"lat".to_string()));
    assert!(var_names.contains(&"lon".to_string()));
    assert!(var_names.contains(&"time".to_string()));
    assert!(var_names.contains(&"temperature".to_string()));

    // Read lat values
    let lat_var = file.variable("lat").unwrap();
    let lat_vals = lat_var.read_raw_f32().unwrap();
    assert_eq!(lat_vals, vec![10.0f32, 20.0, 30.0]);

    // Check CF attributes on temperature
    let mut temp_var = file.variable("temperature").unwrap();
    let cf = temp_var.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("K"));
    assert_eq!(cf.long_name.as_deref(), Some("Air Temperature"));
    assert_eq!(cf.standard_name.as_deref(), Some("air_temperature"));

    // Read temperature data
    let temp_vals = temp_var.read_raw_f32().unwrap();
    assert_eq!(temp_vals.len(), 36); // 3 * 3 * 4
    assert!((temp_vals[0] - 270.0).abs() < 0.01);
    assert!((temp_vals[35] - 305.0).abs() < 0.01);

    // Check global attributes
    let global_attrs = file.global_attrs().unwrap();
    assert!(
        matches!(global_attrs.get("Conventions"), Some(AttrValue::String(s)) if s == "CF-1.8")
    );
    assert!(
        matches!(global_attrs.get("title"), Some(AttrValue::String(s)) if s == "Test CF Dataset")
    );
}

// ===========================================================================
// 2. Python xarray creates file -> rustyhdf5 reads dimensions and variables
// ===========================================================================

#[test]
fn xarray_creates_file_rustyhdf5_reads() {
    skip_if_no_xarray!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("xarray_test.nc");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import xarray as xr
import numpy as np
import pandas as pd

# Create xarray Dataset
times = pd.date_range("2020-01-01", periods=5, freq="D")
lats = [10.0, 20.0, 30.0]
lons = [-120.0, -110.0]

temp = np.random.RandomState(42).randn(5, 3, 2).astype(np.float64) * 10 + 280
precip = np.random.RandomState(123).rand(5, 3, 2).astype(np.float64) * 50

ds = xr.Dataset(
    {{
        "temperature": (["time", "lat", "lon"], temp, {{"units": "K", "long_name": "Temperature"}}),
        "precipitation": (["time", "lat", "lon"], precip, {{"units": "mm/day", "long_name": "Precipitation"}}),
    }},
    coords={{
        "time": times,
        "lat": lats,
        "lon": lons,
    }},
    attrs={{"Conventions": "CF-1.8", "source": "xarray test"}},
)
ds.to_netcdf("{path_str}", engine="netcdf4")
"#
    );
    run_python(&script);

    let file = NetCDF4File::open(&path).unwrap();

    // Check dimensions
    let dims = file.dimensions().unwrap();
    let dim_names: Vec<&str> = dims.iter().map(|d| d.name.as_str()).collect();
    assert!(dim_names.contains(&"lat"));
    assert!(dim_names.contains(&"lon"));
    assert!(dim_names.contains(&"time"));

    let lat_dim = dims.iter().find(|d| d.name == "lat").unwrap();
    assert_eq!(lat_dim.size, 3);

    let lon_dim = dims.iter().find(|d| d.name == "lon").unwrap();
    assert_eq!(lon_dim.size, 2);

    // Check variables exist
    let variables = file.variables().unwrap();
    let var_names: Vec<String> = variables.iter().map(|v| v.name().to_string()).collect();
    assert!(var_names.contains(&"temperature".to_string()));
    assert!(var_names.contains(&"precipitation".to_string()));

    // Read temperature variable
    let mut temp_var = file.variable("temperature").unwrap();
    let shape = temp_var.shape().unwrap();
    assert_eq!(shape, vec![5, 3, 2]); // time=5, lat=3, lon=2
    let temp_vals = temp_var.read_raw_f64().unwrap();
    assert_eq!(temp_vals.len(), 30); // 5*3*2

    // Read precipitation variable
    let precip_var = file.variable("precipitation").unwrap();
    let precip_vals = precip_var.read_raw_f64().unwrap();
    assert_eq!(precip_vals.len(), 30);

    // Check CF attributes
    let cf = temp_var.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("K"));
    assert_eq!(cf.long_name.as_deref(), Some("Temperature"));

    // Check global attributes
    let global_attrs = file.global_attrs().unwrap();
    assert!(
        matches!(global_attrs.get("Conventions"), Some(AttrValue::String(s)) if s == "CF-1.8")
    );
}

// ===========================================================================
// 3. Python netCDF4 creates file with groups -> rustyhdf5 reads groups
// ===========================================================================

#[test]
fn netcdf4_python_creates_grouped_file_rustyhdf5_reads() {
    skip_if_no_netcdf4!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grouped_test.nc");
    let path_str = path.display().to_string();

    let script = format!(
        r#"
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("{path_str}", "w", format="NETCDF4")
ds.title = "Grouped NetCDF4 file"

# Root-level dimension and variable
ds.createDimension("x", 5)
x_var = ds.createVariable("x", "f8", ("x",))
x_var[:] = [1.0, 2.0, 3.0, 4.0, 5.0]

# Group: surface
surface = ds.createGroup("surface")
surface.description = "Surface observations"
surface.createDimension("station", 3)
temp = surface.createVariable("temperature", "f4", ("station",))
temp.units = "K"
temp[:] = [288.0, 290.0, 285.0]

# Group: upper_air
upper = ds.createGroup("upper_air")
upper.description = "Upper air soundings"
upper.createDimension("level", 4)
press = upper.createVariable("pressure", "f4", ("level",))
press.units = "hPa"
press[:] = [1000.0, 850.0, 500.0, 200.0]

ds.close()
"#
    );
    run_python(&script);

    let file = NetCDF4File::open(&path).unwrap();

    // Check root variable
    let x_var = file.variable("x").unwrap();
    let x_vals = x_var.read_raw_f64().unwrap();
    assert_eq!(x_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Check group names
    let group_names = file.group_names().unwrap();
    assert!(group_names.contains(&"surface".to_string()));
    assert!(group_names.contains(&"upper_air".to_string()));

    // Check surface group
    let surface = file.group("surface").unwrap();
    let surface_attrs = surface.attrs().unwrap();
    assert!(matches!(
        surface_attrs.get("description"),
        Some(AttrValue::String(s)) if s == "Surface observations"
    ));

    let surf_vars = surface.variables().unwrap();
    let surf_var_names: Vec<String> = surf_vars.iter().map(|v| v.name().to_string()).collect();
    assert!(surf_var_names.contains(&"temperature".to_string()));

    let temp_var = surface.variable("temperature").unwrap();
    let temp_vals = temp_var.read_raw_f32().unwrap();
    assert_eq!(temp_vals, vec![288.0f32, 290.0, 285.0]);

    // Check upper_air group
    let upper = file.group("upper_air").unwrap();
    let press_var = upper.variable("pressure").unwrap();
    let press_vals = press_var.read_raw_f32().unwrap();
    assert_eq!(press_vals, vec![1000.0f32, 850.0, 500.0, 200.0]);
}
