//! Comprehensive tests for rustyhdf5-netcdf4.
//!
//! We create NetCDF-4 style HDF5 files using rustyhdf5's FileBuilder with the
//! appropriate NetCDF-4 conventions (CLASS=DIMENSION_SCALE, _Netcdf4Dimid,
//! _NCProperties, CF attributes, etc.).

use rustyhdf5::{AttrValue, FileBuilder};
use rustyhdf5_netcdf4::{Dimension, FillValue, NcType, NetCDF4File};

// ---------------------------------------------------------------------------
// Helpers: build NetCDF-4 style HDF5 files in memory
// ---------------------------------------------------------------------------

/// Create a simple NetCDF-4 file with lat/lon dimensions and a temperature variable.
fn make_simple_netcdf4() -> Vec<u8> {
    let mut b = FileBuilder::new();

    // Root attributes (NetCDF-4 provenance)
    b.set_attr(
        "_NCProperties",
        AttrValue::String("version=2,rustyhdf5=0.1.0".into()),
    );
    b.set_attr(
        "Conventions",
        AttrValue::String("CF-1.8".into()),
    );
    b.set_attr(
        "history",
        AttrValue::String("created by rustyhdf5-netcdf4 test".into()),
    );

    // Dimension: lat (3 values)
    b.create_dataset("lat")
        .with_f64_data(&[-30.0, 0.0, 30.0])
        .with_shape(&[3])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0))
        .set_attr("units", AttrValue::String("degrees_north".into()))
        .set_attr("long_name", AttrValue::String("latitude".into()))
        .set_attr("standard_name", AttrValue::String("latitude".into()))
        .set_attr("axis", AttrValue::String("Y".into()));

    // Dimension: lon (4 values)
    b.create_dataset("lon")
        .with_f64_data(&[0.0, 90.0, 180.0, 270.0])
        .with_shape(&[4])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(1))
        .set_attr("units", AttrValue::String("degrees_east".into()))
        .set_attr("long_name", AttrValue::String("longitude".into()))
        .set_attr("standard_name", AttrValue::String("longitude".into()))
        .set_attr("axis", AttrValue::String("X".into()));

    // Variable: temperature (3x4)
    b.create_dataset("temperature")
        .with_f64_data(&[
            20.0, 22.0, 25.0, 23.0,
            28.0, 30.0, 29.0, 27.0,
            15.0, 18.0, 20.0, 17.0,
        ])
        .with_shape(&[3, 4])
        .set_attr("units", AttrValue::String("K".into()))
        .set_attr("long_name", AttrValue::String("Surface Temperature".into()))
        .set_attr(
            "standard_name",
            AttrValue::String("surface_temperature".into()),
        )
        .set_attr("_FillValue", AttrValue::F64(-9999.0));

    b.finish().unwrap()
}

/// Create a NetCDF-4 file with scale_factor and add_offset for packed data.
fn make_packed_netcdf4() -> Vec<u8> {
    let mut b = FileBuilder::new();

    b.set_attr(
        "_NCProperties",
        AttrValue::String("version=2,rustyhdf5=0.1.0".into()),
    );

    // Dimension: time (5 values)
    b.create_dataset("time")
        .with_f64_data(&[0.0, 1.0, 2.0, 3.0, 4.0])
        .with_shape(&[5])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0))
        .set_attr("units", AttrValue::String("days since 2000-01-01".into()))
        .set_attr("calendar", AttrValue::String("standard".into()))
        .set_attr("axis", AttrValue::String("T".into()));

    // Variable: packed_temp with scale_factor and add_offset
    // Unpacked = packed * 0.01 + 273.15
    b.create_dataset("packed_temp")
        .with_f64_data(&[100.0, 200.0, -9999.0, 300.0, 150.0])
        .with_shape(&[5])
        .set_attr("scale_factor", AttrValue::F64(0.01))
        .set_attr("add_offset", AttrValue::F64(273.15))
        .set_attr("_FillValue", AttrValue::F64(-9999.0))
        .set_attr("missing_value", AttrValue::F64(-9999.0))
        .set_attr("units", AttrValue::String("K".into()))
        .set_attr("long_name", AttrValue::String("Packed Temperature".into()));

    b.finish().unwrap()
}

/// Create a NetCDF-4 file with groups.
fn make_grouped_netcdf4() -> Vec<u8> {
    let mut b = FileBuilder::new();

    b.set_attr(
        "_NCProperties",
        AttrValue::String("version=2,rustyhdf5=0.1.0".into()),
    );
    b.set_attr("title", AttrValue::String("Grouped NetCDF-4".into()));

    // Root-level dimension
    b.create_dataset("time")
        .with_f64_data(&[0.0, 1.0, 2.0])
        .with_shape(&[3])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0))
        .set_attr("units", AttrValue::String("hours since 2020-01-01".into()));

    // Group: surface
    let mut g_surface = b.create_group("surface");
    g_surface
        .create_dataset("pressure")
        .with_f64_data(&[1013.25, 1012.0, 1011.5])
        .set_attr("units", AttrValue::String("hPa".into()))
        .set_attr(
            "long_name",
            AttrValue::String("Surface Pressure".into()),
        );
    g_surface.set_attr("description", AttrValue::String("Surface variables".into()));
    let finished_surface = g_surface.finish();
    b.add_group(finished_surface);

    // Group: upper_air
    let mut g_upper = b.create_group("upper_air");
    g_upper
        .create_dataset("wind_speed")
        .with_f64_data(&[5.0, 7.5, 6.2])
        .set_attr("units", AttrValue::String("m/s".into()))
        .set_attr(
            "long_name",
            AttrValue::String("Wind Speed at 500hPa".into()),
        );
    let finished_upper = g_upper.finish();
    b.add_group(finished_upper);

    b.finish().unwrap()
}

/// Create a file with various data types.
fn make_multitype_netcdf4() -> Vec<u8> {
    let mut b = FileBuilder::new();

    // Dimension
    b.create_dataset("x")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .with_shape(&[3])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0));

    // Float64 variable
    b.create_dataset("var_f64")
        .with_f64_data(&[1.1, 2.2, 3.3])
        .with_shape(&[3]);

    // Float32 variable
    b.create_dataset("var_f32")
        .with_f32_data(&[1.5f32, 2.5, 3.5])
        .with_shape(&[3]);

    // Int32 variable
    b.create_dataset("var_i32")
        .with_i32_data(&[10, 20, 30])
        .with_shape(&[3]);

    // Int64 variable
    b.create_dataset("var_i64")
        .with_i64_data(&[100, 200, 300])
        .with_shape(&[3]);

    // UInt8 variable
    b.create_dataset("var_u8")
        .with_u8_data(&[1, 2, 3])
        .with_shape(&[3]);

    b.finish().unwrap()
}

/// Create a file with valid_range and additional CF attributes.
fn make_cf_rich_netcdf4() -> Vec<u8> {
    let mut b = FileBuilder::new();

    b.create_dataset("level")
        .with_f64_data(&[1000.0, 850.0, 500.0, 200.0])
        .with_shape(&[4])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0))
        .set_attr("units", AttrValue::String("hPa".into()))
        .set_attr(
            "long_name",
            AttrValue::String("Pressure Level".into()),
        )
        .set_attr("axis", AttrValue::String("Z".into()))
        .set_attr(
            "valid_range",
            AttrValue::F64Array(vec![0.0, 1100.0]),
        );

    b.finish().unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_open_netcdf4_from_bytes() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    assert!(format!("{file:?}").contains("NetCDF4File"));
}

#[test]
fn test_nc_properties() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let props = file.nc_properties().unwrap();
    assert!(props.is_some());
    assert!(props.unwrap().contains("rustyhdf5"));
}

#[test]
fn test_global_attrs() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let attrs = file.global_attrs().unwrap();

    assert!(matches!(
        attrs.get("Conventions"),
        Some(AttrValue::String(s)) if s == "CF-1.8"
    ));
    assert!(matches!(
        attrs.get("history"),
        Some(AttrValue::String(s)) if s.contains("rustyhdf5-netcdf4")
    ));
}

#[test]
fn test_read_dimensions() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let dims = file.dimensions().unwrap();

    assert_eq!(dims.len(), 2);

    let lat_dim = dims.iter().find(|d| d.name == "lat").unwrap();
    assert_eq!(lat_dim.size, 3);
    assert!(!lat_dim.is_unlimited);

    let lon_dim = dims.iter().find(|d| d.name == "lon").unwrap();
    assert_eq!(lon_dim.size, 4);
    assert!(!lon_dim.is_unlimited);
}

#[test]
fn test_dimension_order_by_dimid() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let dims = file.dimensions().unwrap();

    // Dimensions should be sorted by _Netcdf4Dimid: lat=0, lon=1
    assert_eq!(dims[0].name, "lat");
    assert_eq!(dims[1].name, "lon");
}

#[test]
fn test_read_variables() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let vars = file.variables().unwrap();

    // Should have lat, lon, temperature
    assert_eq!(vars.len(), 3);
    let var_names: Vec<&str> = vars.iter().map(|v| v.name()).collect();
    assert!(var_names.contains(&"lat"));
    assert!(var_names.contains(&"lon"));
    assert!(var_names.contains(&"temperature"));
}

#[test]
fn test_variable_dimensions() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut temp = file.variable("temperature").unwrap();

    let dims = temp.dimensions();
    assert_eq!(dims.len(), 2);
    assert_eq!(dims[0].size, 3); // lat
    assert_eq!(dims[1].size, 4); // lon

    let shape = temp.shape().unwrap();
    assert_eq!(shape, vec![3, 4]);

    let _ = temp.attrs().unwrap();
}

#[test]
fn test_variable_cf_attributes() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut temp = file.variable("temperature").unwrap();

    let cf = temp.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("K"));
    assert_eq!(cf.long_name.as_deref(), Some("Surface Temperature"));
    assert_eq!(cf.standard_name.as_deref(), Some("surface_temperature"));
    assert_eq!(cf.fill_value, Some(FillValue::Float(-9999.0)));
    assert!(cf.scale_factor.is_none());
    assert!(cf.add_offset.is_none());
}

#[test]
fn test_coordinate_variable() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let lat = file.variable("lat").unwrap();

    // lat is a coordinate variable: 1-D with name matching its dimension
    assert!(lat.is_coordinate());

    let temp = file.variable("temperature").unwrap();
    assert!(!temp.is_coordinate());
}

#[test]
fn test_coordinate_cf_attrs() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut lat = file.variable("lat").unwrap();

    let cf = lat.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("degrees_north"));
    assert_eq!(cf.axis.as_deref(), Some("Y"));
    assert_eq!(cf.standard_name.as_deref(), Some("latitude"));
}

#[test]
fn test_read_raw_f64() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let lat = file.variable("lat").unwrap();
    let data = lat.read_raw_f64().unwrap();
    assert_eq!(data, vec![-30.0, 0.0, 30.0]);
}

#[test]
fn test_read_temperature_data() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut temp = file.variable("temperature").unwrap();
    let data = temp.read_f64().unwrap();

    // No scale_factor/add_offset, so data should be unchanged
    assert_eq!(data.len(), 12);
    assert_eq!(data[0], 20.0);
    assert_eq!(data[4], 28.0);
}

#[test]
fn test_scale_factor_add_offset() {
    let bytes = make_packed_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut var = file.variable("packed_temp").unwrap();

    let cf = var.cf_attributes().unwrap();
    assert_eq!(cf.scale_factor, Some(0.01));
    assert_eq!(cf.add_offset, Some(273.15));

    let data = var.read_f64().unwrap();
    assert_eq!(data.len(), 5);

    // packed=100 -> 100*0.01 + 273.15 = 274.15
    assert!((data[0] - 274.15).abs() < 1e-10);
    // packed=200 -> 200*0.01 + 273.15 = 275.15
    assert!((data[1] - 275.15).abs() < 1e-10);
    // packed=-9999 (fill) -> NaN
    assert!(data[2].is_nan());
    // packed=300 -> 300*0.01 + 273.15 = 276.15
    assert!((data[3] - 276.15).abs() < 1e-10);
    // packed=150 -> 150*0.01 + 273.15 = 274.65
    assert!((data[4] - 274.65).abs() < 1e-10);
}

#[test]
fn test_raw_read_bypasses_scaling() {
    let bytes = make_packed_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("packed_temp").unwrap();

    let raw = var.read_raw_f64().unwrap();
    assert_eq!(raw, vec![100.0, 200.0, -9999.0, 300.0, 150.0]);
}

#[test]
fn test_groups() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    let mut group_names = file.group_names().unwrap();
    group_names.sort();
    assert_eq!(group_names, vec!["surface", "upper_air"]);
}

#[test]
fn test_group_attrs() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    let surface = file.group("surface").unwrap();
    let attrs = surface.attrs().unwrap();
    assert!(matches!(
        attrs.get("description"),
        Some(AttrValue::String(s)) if s == "Surface variables"
    ));
}

#[test]
fn test_group_variables() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    let surface = file.group("surface").unwrap();
    let var_names = surface.variable_names().unwrap();
    assert!(var_names.contains(&"pressure".to_string()));

    let mut pressure = surface.variable("pressure").unwrap();
    let data = pressure.read_f64().unwrap();
    assert_eq!(data, vec![1013.25, 1012.0, 1011.5]);

    let cf = pressure.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("hPa"));
    assert_eq!(cf.long_name.as_deref(), Some("Surface Pressure"));
}

#[test]
fn test_group_nested_access() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    let upper = file.group("upper_air").unwrap();
    let mut wind = upper.variable("wind_speed").unwrap();
    let data = wind.read_f64().unwrap();
    assert_eq!(data, vec![5.0, 7.5, 6.2]);

    let cf = wind.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("m/s"));
}

#[test]
fn test_nc_type_f64() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_f64").unwrap();
    assert_eq!(var.nc_type().unwrap(), NcType::Double);
}

#[test]
fn test_nc_type_f32() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_f32").unwrap();
    assert_eq!(var.nc_type().unwrap(), NcType::Float);
}

#[test]
fn test_nc_type_i32() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_i32").unwrap();
    assert_eq!(var.nc_type().unwrap(), NcType::Int);
}

#[test]
fn test_nc_type_i64() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_i64").unwrap();
    assert_eq!(var.nc_type().unwrap(), NcType::Int64);
}

#[test]
fn test_nc_type_u8() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_u8").unwrap();
    assert_eq!(var.nc_type().unwrap(), NcType::UByte);
}

#[test]
fn test_read_i32_data() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_i32").unwrap();
    let data = var.read_raw_i32().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

#[test]
fn test_read_f32_data() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_f32").unwrap();
    let data = var.read_raw_f32().unwrap();
    assert_eq!(data, vec![1.5f32, 2.5, 3.5]);
}

#[test]
fn test_read_i64_data() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_i64").unwrap();
    let data = var.read_raw_i64().unwrap();
    assert_eq!(data, vec![100, 200, 300]);
}

#[test]
fn test_valid_range() {
    let bytes = make_cf_rich_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut var = file.variable("level").unwrap();
    let cf = var.cf_attributes().unwrap();
    assert_eq!(cf.valid_range, Some((0.0, 1100.0)));
    assert_eq!(cf.axis.as_deref(), Some("Z"));
}

#[test]
fn test_nc_type_display() {
    assert_eq!(NcType::Double.to_string(), "NC_DOUBLE");
    assert_eq!(NcType::Float.to_string(), "NC_FLOAT");
    assert_eq!(NcType::Int.to_string(), "NC_INT");
    assert_eq!(NcType::Byte.to_string(), "NC_BYTE");
    assert_eq!(NcType::String.to_string(), "NC_STRING");
}

#[test]
fn test_variable_not_found_error() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let err = file.variable("nonexistent").unwrap_err();
    assert!(err.to_string().contains("variable not found"));
}

#[test]
fn test_group_not_found_error() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let err = file.group("nonexistent").unwrap_err();
    assert!(err.to_string().contains("group not found"));
}

#[test]
fn test_error_display() {
    use rustyhdf5_netcdf4::Error;

    let err = Error::DimensionNotFound("time".into());
    assert_eq!(err.to_string(), "dimension not found: time");

    let err = Error::TypeError("cannot convert".into());
    assert_eq!(err.to_string(), "type error: cannot convert");

    let err = Error::NotNetCDF4("missing signature".into());
    assert!(err.to_string().contains("not a NetCDF-4 file"));
}

#[test]
fn test_fill_value_as_f64() {
    assert_eq!(FillValue::Float(1.0).as_f64(), Some(1.0));
    assert_eq!(FillValue::Int(-999).as_f64(), Some(-999.0));
    assert_eq!(FillValue::UInt(42).as_f64(), Some(42.0));
    assert_eq!(FillValue::String("na".into()).as_f64(), None);
}

#[test]
fn test_time_dimension_with_calendar() {
    let bytes = make_packed_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    let mut time_var = file.variable("time").unwrap();
    let cf = time_var.cf_attributes().unwrap();
    assert_eq!(cf.units.as_deref(), Some("days since 2000-01-01"));
    assert_eq!(cf.calendar.as_deref(), Some("standard"));
    assert_eq!(cf.axis.as_deref(), Some("T"));
}

#[test]
fn test_open_from_disk() {
    let bytes = make_simple_netcdf4();
    let path = std::env::temp_dir().join("rustyhdf5_netcdf4_test.nc");
    std::fs::write(&path, &bytes).unwrap();

    let file = NetCDF4File::open(&path).unwrap();
    let dims = file.dimensions().unwrap();
    assert_eq!(dims.len(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_global_attrs_title() {
    let bytes = make_grouped_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let attrs = file.global_attrs().unwrap();
    assert!(matches!(
        attrs.get("title"),
        Some(AttrValue::String(s)) if s == "Grouped NetCDF-4"
    ));
}

#[test]
fn test_real_world_temperature_grid() {
    // Simulate a real-world temperature dataset with lat/lon/time
    let mut b = FileBuilder::new();

    b.set_attr("Conventions", AttrValue::String("CF-1.8".into()));
    b.set_attr(
        "_NCProperties",
        AttrValue::String("version=2,rustyhdf5=0.1.0".into()),
    );
    b.set_attr(
        "title",
        AttrValue::String("ERA5 Temperature Reanalysis".into()),
    );
    b.set_attr("institution", AttrValue::String("ECMWF".into()));

    // Time dimension (2 steps)
    b.create_dataset("time")
        .with_f64_data(&[0.0, 6.0])
        .with_shape(&[2])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(0))
        .set_attr("units", AttrValue::String("hours since 2020-01-01".into()))
        .set_attr("calendar", AttrValue::String("proleptic_gregorian".into()))
        .set_attr("axis", AttrValue::String("T".into()));

    // Latitude dimension (3 values)
    b.create_dataset("latitude")
        .with_f64_data(&[90.0, 0.0, -90.0])
        .with_shape(&[3])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(1))
        .set_attr("units", AttrValue::String("degrees_north".into()))
        .set_attr("axis", AttrValue::String("Y".into()));

    // Longitude dimension (4 values)
    b.create_dataset("longitude")
        .with_f64_data(&[0.0, 90.0, 180.0, 270.0])
        .with_shape(&[4])
        .set_attr("CLASS", AttrValue::String("DIMENSION_SCALE".into()))
        .set_attr("_Netcdf4Dimid", AttrValue::I64(2))
        .set_attr("units", AttrValue::String("degrees_east".into()))
        .set_attr("axis", AttrValue::String("X".into()));

    // Temperature variable (2x3x4) with scale/offset
    let temp_data: Vec<f64> = (0..24).map(|i| i as f64 * 100.0).collect();
    b.create_dataset("t2m")
        .with_f64_data(&temp_data)
        .with_shape(&[2, 3, 4])
        .set_attr("units", AttrValue::String("K".into()))
        .set_attr(
            "long_name",
            AttrValue::String("2 metre temperature".into()),
        )
        .set_attr(
            "standard_name",
            AttrValue::String("air_temperature".into()),
        )
        .set_attr("scale_factor", AttrValue::F64(0.001))
        .set_attr("add_offset", AttrValue::F64(273.15))
        .set_attr("_FillValue", AttrValue::F64(-32767.0));

    let bytes = b.finish().unwrap();
    let file = NetCDF4File::from_bytes(bytes).unwrap();

    // Check dimensions
    let dims = file.dimensions().unwrap();
    assert_eq!(dims.len(), 3);
    assert_eq!(dims[0].name, "time");
    assert_eq!(dims[0].size, 2);
    assert_eq!(dims[1].name, "latitude");
    assert_eq!(dims[1].size, 3);
    assert_eq!(dims[2].name, "longitude");
    assert_eq!(dims[2].size, 4);

    // Check variables
    let vars = file.variables().unwrap();
    assert_eq!(vars.len(), 4); // time, latitude, longitude, t2m

    // Check temperature with scaling
    let mut t2m = file.variable("t2m").unwrap();
    let cf = t2m.cf_attributes().unwrap();
    assert_eq!(cf.scale_factor, Some(0.001));
    assert_eq!(cf.add_offset, Some(273.15));
    assert_eq!(cf.standard_name.as_deref(), Some("air_temperature"));

    let data = t2m.read_f64().unwrap();
    assert_eq!(data.len(), 24);
    // First value: 0.0 * 0.001 + 273.15 = 273.15
    assert!((data[0] - 273.15).abs() < 1e-10);
    // Second value: 100.0 * 0.001 + 273.15 = 273.25
    assert!((data[1] - 273.25).abs() < 1e-10);

    // Check global attrs
    let attrs = file.global_attrs().unwrap();
    assert!(matches!(
        attrs.get("institution"),
        Some(AttrValue::String(s)) if s == "ECMWF"
    ));
}

#[test]
fn test_variable_read_raw_bytes() {
    let bytes = make_multitype_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let var = file.variable("var_i32").unwrap();
    let raw = var.read_raw().unwrap();
    // 3 i32 values = 12 bytes
    assert_eq!(raw.len(), 12);
    // First i32 = 10
    let val = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    assert_eq!(val, 10);
}

#[test]
fn test_hdf5_file_access() {
    let bytes = make_simple_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let hdf5 = file.hdf5_file();
    // Can use the underlying HDF5 file for advanced operations
    assert!(hdf5.superblock().version >= 2);
}

#[test]
fn test_missing_value_attribute() {
    let bytes = make_packed_netcdf4();
    let file = NetCDF4File::from_bytes(bytes).unwrap();
    let mut var = file.variable("packed_temp").unwrap();
    let cf = var.cf_attributes().unwrap();
    assert_eq!(cf.missing_value, Some(FillValue::Float(-9999.0)));
}

#[test]
fn test_dimension_struct_equality() {
    let d1 = Dimension {
        name: "time".into(),
        size: 10,
        is_unlimited: false,
    };
    let d2 = Dimension {
        name: "time".into(),
        size: 10,
        is_unlimited: false,
    };
    assert_eq!(d1, d2);

    let d3 = Dimension {
        name: "time".into(),
        size: 10,
        is_unlimited: true,
    };
    assert_ne!(d1, d3);
}
