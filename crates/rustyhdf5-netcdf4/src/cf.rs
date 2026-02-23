//! CF (Climate and Forecast) convention attribute handling.
//!
//! This module reads and exposes standard CF attributes from NetCDF-4 variables:
//! `units`, `long_name`, `standard_name`, `_FillValue`, `missing_value`,
//! `scale_factor`, `add_offset`, `valid_range`, `calendar`, `axis`.

use std::collections::HashMap;

use rustyhdf5::AttrValue;

/// Standard CF attributes extracted from a NetCDF-4 variable.
#[derive(Debug, Clone, Default)]
pub struct CfAttributes {
    /// Physical units of the variable (e.g., "K", "m/s", "degrees_east").
    pub units: Option<String>,
    /// Human-readable name for the variable.
    pub long_name: Option<String>,
    /// CF standard name (e.g., "air_temperature").
    pub standard_name: Option<String>,
    /// Fill value indicating missing data in the native type.
    pub fill_value: Option<FillValue>,
    /// Alternative missing value indicator.
    pub missing_value: Option<FillValue>,
    /// Scale factor for packed data: `unpacked = packed * scale_factor + add_offset`.
    pub scale_factor: Option<f64>,
    /// Offset for packed data: `unpacked = packed * scale_factor + add_offset`.
    pub add_offset: Option<f64>,
    /// Valid range `[min, max]` for the variable data.
    pub valid_range: Option<(f64, f64)>,
    /// Calendar type for time variables (e.g., "standard", "365_day").
    pub calendar: Option<String>,
    /// Axis designation: "X", "Y", "Z", or "T".
    pub axis: Option<String>,
}

/// A fill/missing value that can be numeric or string.
#[derive(Debug, Clone, PartialEq)]
pub enum FillValue {
    /// Floating-point fill value.
    Float(f64),
    /// Integer fill value.
    Int(i64),
    /// Unsigned integer fill value.
    UInt(u64),
    /// String fill value.
    String(String),
}

impl FillValue {
    /// Convert to f64 if possible.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FillValue::Float(v) => Some(*v),
            FillValue::Int(v) => Some(*v as f64),
            FillValue::UInt(v) => Some(*v as f64),
            FillValue::String(_) => None,
        }
    }
}

/// Extract CF attributes from a HashMap of HDF5 attributes.
pub fn extract_cf_attributes(attrs: &HashMap<String, AttrValue>) -> CfAttributes {
    CfAttributes {
        units: get_string(attrs, "units"),
        long_name: get_string(attrs, "long_name"),
        standard_name: get_string(attrs, "standard_name"),
        fill_value: get_fill_value(attrs, "_FillValue"),
        missing_value: get_fill_value(attrs, "missing_value"),
        scale_factor: get_f64(attrs, "scale_factor"),
        add_offset: get_f64(attrs, "add_offset"),
        valid_range: get_valid_range(attrs),
        calendar: get_string(attrs, "calendar"),
        axis: get_string(attrs, "axis"),
    }
}

/// Apply scale_factor and add_offset to raw f64 data.
///
/// Formula: `unpacked = packed * scale_factor + add_offset`
///
/// If neither scale_factor nor add_offset are present, returns the data unchanged.
/// Missing values (matching fill_value or missing_value) are preserved as NaN.
pub fn apply_scale_offset(
    data: &[f64],
    cf: &CfAttributes,
) -> Vec<f64> {
    let scale = cf.scale_factor.unwrap_or(1.0);
    let offset = cf.add_offset.unwrap_or(0.0);
    let has_transform = cf.scale_factor.is_some() || cf.add_offset.is_some();

    if !has_transform {
        return data.to_vec();
    }

    let fill_f64 = cf.fill_value.as_ref().and_then(|fv| fv.as_f64());
    let missing_f64 = cf.missing_value.as_ref().and_then(|fv| fv.as_f64());

    data.iter()
        .map(|&v| {
            if is_missing(v, fill_f64, missing_f64) {
                f64::NAN
            } else {
                v * scale + offset
            }
        })
        .collect()
}

/// Check if a value matches the fill value or missing value.
fn is_missing(value: f64, fill: Option<f64>, missing: Option<f64>) -> bool {
    if let Some(fv) = fill {
        if (value - fv).abs() < f64::EPSILON || (value.is_nan() && fv.is_nan()) {
            return true;
        }
    }
    if let Some(mv) = missing {
        if (value - mv).abs() < f64::EPSILON || (value.is_nan() && mv.is_nan()) {
            return true;
        }
    }
    false
}

fn get_string(attrs: &HashMap<String, AttrValue>, key: &str) -> Option<String> {
    match attrs.get(key) {
        Some(AttrValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn get_f64(attrs: &HashMap<String, AttrValue>, key: &str) -> Option<f64> {
    match attrs.get(key) {
        Some(AttrValue::F64(v)) => Some(*v),
        Some(AttrValue::I64(v)) => Some(*v as f64),
        Some(AttrValue::U64(v)) => Some(*v as f64),
        _ => None,
    }
}

fn get_fill_value(attrs: &HashMap<String, AttrValue>, key: &str) -> Option<FillValue> {
    match attrs.get(key) {
        Some(AttrValue::F64(v)) => Some(FillValue::Float(*v)),
        Some(AttrValue::I64(v)) => Some(FillValue::Int(*v)),
        Some(AttrValue::U64(v)) => Some(FillValue::UInt(*v)),
        Some(AttrValue::String(s)) => Some(FillValue::String(s.clone())),
        Some(AttrValue::F64Array(arr)) if !arr.is_empty() => Some(FillValue::Float(arr[0])),
        Some(AttrValue::I64Array(arr)) if !arr.is_empty() => Some(FillValue::Int(arr[0])),
        _ => None,
    }
}

fn get_valid_range(attrs: &HashMap<String, AttrValue>) -> Option<(f64, f64)> {
    // Try valid_range attribute (2-element array)
    match attrs.get("valid_range") {
        Some(AttrValue::F64Array(arr)) if arr.len() >= 2 => {
            return Some((arr[0], arr[1]));
        }
        Some(AttrValue::I64Array(arr)) if arr.len() >= 2 => {
            return Some((arr[0] as f64, arr[1] as f64));
        }
        _ => {}
    }

    // Fallback: try valid_min and valid_max separately
    let min = get_f64(attrs, "valid_min");
    let max = get_f64(attrs, "valid_max");
    match (min, max) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_cf_empty() {
        let attrs = HashMap::new();
        let cf = extract_cf_attributes(&attrs);
        assert!(cf.units.is_none());
        assert!(cf.scale_factor.is_none());
        assert!(cf.add_offset.is_none());
    }

    #[test]
    fn test_extract_cf_full() {
        let mut attrs = HashMap::new();
        attrs.insert("units".into(), AttrValue::String("K".into()));
        attrs.insert("long_name".into(), AttrValue::String("Temperature".into()));
        attrs.insert(
            "standard_name".into(),
            AttrValue::String("air_temperature".into()),
        );
        attrs.insert("_FillValue".into(), AttrValue::F64(-9999.0));
        attrs.insert("scale_factor".into(), AttrValue::F64(0.01));
        attrs.insert("add_offset".into(), AttrValue::F64(273.15));
        attrs.insert("calendar".into(), AttrValue::String("standard".into()));
        attrs.insert("axis".into(), AttrValue::String("T".into()));

        let cf = extract_cf_attributes(&attrs);
        assert_eq!(cf.units.as_deref(), Some("K"));
        assert_eq!(cf.long_name.as_deref(), Some("Temperature"));
        assert_eq!(cf.standard_name.as_deref(), Some("air_temperature"));
        assert_eq!(cf.fill_value, Some(FillValue::Float(-9999.0)));
        assert_eq!(cf.scale_factor, Some(0.01));
        assert_eq!(cf.add_offset, Some(273.15));
        assert_eq!(cf.calendar.as_deref(), Some("standard"));
        assert_eq!(cf.axis.as_deref(), Some("T"));
    }

    #[test]
    fn test_apply_scale_offset_no_transform() {
        let data = vec![1.0, 2.0, 3.0];
        let cf = CfAttributes::default();
        let result = apply_scale_offset(&data, &cf);
        assert_eq!(result, data);
    }

    #[test]
    fn test_apply_scale_offset_with_transform() {
        let data = vec![100.0, 200.0, 300.0];
        let cf = CfAttributes {
            scale_factor: Some(0.01),
            add_offset: Some(273.15),
            ..Default::default()
        };
        let result = apply_scale_offset(&data, &cf);
        assert!((result[0] - 274.15).abs() < 1e-10);
        assert!((result[1] - 275.15).abs() < 1e-10);
        assert!((result[2] - 276.15).abs() < 1e-10);
    }

    #[test]
    fn test_apply_scale_offset_with_fill() {
        let data = vec![100.0, -9999.0, 300.0];
        let cf = CfAttributes {
            scale_factor: Some(0.01),
            add_offset: Some(273.15),
            fill_value: Some(FillValue::Float(-9999.0)),
            ..Default::default()
        };
        let result = apply_scale_offset(&data, &cf);
        assert!((result[0] - 274.15).abs() < 1e-10);
        assert!(result[1].is_nan());
        assert!((result[2] - 276.15).abs() < 1e-10);
    }

    #[test]
    fn test_valid_range_array() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "valid_range".into(),
            AttrValue::F64Array(vec![0.0, 100.0]),
        );
        let cf = extract_cf_attributes(&attrs);
        assert_eq!(cf.valid_range, Some((0.0, 100.0)));
    }

    #[test]
    fn test_valid_range_min_max() {
        let mut attrs = HashMap::new();
        attrs.insert("valid_min".into(), AttrValue::F64(-10.0));
        attrs.insert("valid_max".into(), AttrValue::F64(50.0));
        let cf = extract_cf_attributes(&attrs);
        assert_eq!(cf.valid_range, Some((-10.0, 50.0)));
    }

    #[test]
    fn test_fill_value_int() {
        let mut attrs = HashMap::new();
        attrs.insert("_FillValue".into(), AttrValue::I64(-999));
        let cf = extract_cf_attributes(&attrs);
        assert_eq!(cf.fill_value, Some(FillValue::Int(-999)));
        assert_eq!(cf.fill_value.unwrap().as_f64(), Some(-999.0));
    }
}
