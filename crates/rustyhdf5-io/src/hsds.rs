//! HSDS (HDF5 Highly Scalable Data Service) REST API client.
//!
//! Provides an async HTTP-based client for reading HDF5 data from an HSDS server.
//! Gated behind the `hsds` feature flag.

use std::io;

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Error type for HSDS operations.
#[derive(Debug)]
pub enum HsdsError {
    /// HTTP request error.
    Http(reqwest::Error),
    /// JSON parsing error.
    Json(serde_json::Error),
    /// Server returned an error response.
    Server { status: u16, message: String },
    /// Data conversion error.
    DataError(String),
    /// I/O error.
    Io(io::Error),
}

impl std::fmt::Display for HsdsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HsdsError::Http(e) => write!(f, "HTTP error: {e}"),
            HsdsError::Json(e) => write!(f, "JSON error: {e}"),
            HsdsError::Server { status, message } => {
                write!(f, "server error (HTTP {status}): {message}")
            }
            HsdsError::DataError(msg) => write!(f, "data error: {msg}"),
            HsdsError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for HsdsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HsdsError::Http(e) => Some(e),
            HsdsError::Json(e) => Some(e),
            HsdsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for HsdsError {
    fn from(e: reqwest::Error) -> Self {
        HsdsError::Http(e)
    }
}

impl From<serde_json::Error> for HsdsError {
    fn from(e: serde_json::Error) -> Self {
        HsdsError::Json(e)
    }
}

impl From<io::Error> for HsdsError {
    fn from(e: io::Error) -> Self {
        HsdsError::Io(e)
    }
}

/// Information about an HSDS domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainInfo {
    /// The root group ID.
    pub root: String,
    /// Domain creation timestamp (seconds since epoch).
    #[serde(default)]
    pub created: f64,
    /// Domain last-modified timestamp.
    #[serde(default, rename = "lastModified")]
    pub last_modified: f64,
    /// Owner of the domain.
    #[serde(default)]
    pub owner: String,
}

/// Information about a link within a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkInfo {
    /// Name of the link.
    pub title: String,
    /// Link class: "H5L_TYPE_HARD", "H5L_TYPE_SOFT", "H5L_TYPE_EXTERNAL".
    pub class: String,
    /// Target object ID (for hard links).
    #[serde(default)]
    pub id: Option<String>,
    /// Target path (for soft links).
    #[serde(default)]
    pub h5path: Option<String>,
    /// Target of the link.
    #[serde(default)]
    pub target: Option<LinkTarget>,
}

/// Target information for a link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkTarget {
    /// Object ID of the target.
    #[serde(default)]
    pub id: Option<String>,
    /// Collection type: "groups", "datasets", "datatypes".
    #[serde(default)]
    pub collection: Option<String>,
}

/// Information about an HSDS dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfoHsds {
    /// Dataset ID.
    pub id: String,
    /// Shape information.
    pub shape: ShapeInfo,
    /// Type information.
    #[serde(rename = "type")]
    pub dtype: TypeInfo,
}

/// Shape information from HSDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeInfo {
    /// Shape class: "H5S_SIMPLE", "H5S_SCALAR", "H5S_NULL".
    pub class: String,
    /// Dimensions (for simple shapes).
    #[serde(default)]
    pub dims: Vec<u64>,
    /// Maximum dimensions (may contain 0 for unlimited).
    #[serde(default)]
    pub maxdims: Vec<u64>,
}

/// Type information from HSDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type class: "H5T_FLOAT", "H5T_INTEGER", "H5T_STRING", etc.
    pub class: String,
    /// Base type (for predefined types like "H5T_IEEE_F64LE").
    #[serde(default)]
    pub base: Option<String>,
}

/// HSDS value response wrapper.
#[derive(Debug, Clone, Deserialize)]
struct ValueResponse {
    value: serde_json::Value,
}

/// Response from the links endpoint.
#[derive(Debug, Clone, Deserialize)]
struct LinksResponse {
    links: Vec<LinkInfo>,
}

/// Client for interacting with an HSDS REST API server.
///
/// HSDS (HDF5 Highly Scalable Data Service) exposes HDF5 data over a RESTful
/// HTTP API. This client provides async methods for querying domains, groups,
/// datasets, and reading values.
#[derive(Debug, Clone)]
pub struct HsdsClient {
    base_url: String,
    client: Client,
}

impl HsdsClient {
    /// Create a new HSDS client pointing at the given base URL.
    ///
    /// The URL should include the protocol and host, e.g. `"http://localhost:5101"`.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    /// Create a new client with a custom reqwest `Client`.
    ///
    /// Useful for testing with mock transports or custom TLS configuration.
    pub fn with_client(base_url: &str, client: Client) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Get information about an HSDS domain.
    ///
    /// The domain is specified as a path like `"/home/user/myfile.h5"`.
    pub async fn get_domain(&self, domain: &str) -> Result<DomainInfo, HsdsError> {
        let url = format!("{}/?host={}", self.base_url, domain);
        let resp = self.client.get(&url).send().await?;
        check_status(&resp)?;
        let body: DomainInfo = resp.json().await?;
        Ok(body)
    }

    /// List links (children) of a group within a domain.
    pub async fn list_groups(
        &self,
        domain: &str,
        group_id: &str,
    ) -> Result<Vec<LinkInfo>, HsdsError> {
        let url = format!(
            "{}/groups/{}/links?host={}",
            self.base_url, group_id, domain
        );
        let resp = self.client.get(&url).send().await?;
        check_status(&resp)?;
        let body: LinksResponse = resp.json().await?;
        Ok(body.links)
    }

    /// Get information about a dataset.
    pub async fn get_dataset(
        &self,
        domain: &str,
        dataset_id: &str,
    ) -> Result<DatasetInfoHsds, HsdsError> {
        let url = format!(
            "{}/datasets/{}?host={}",
            self.base_url, dataset_id, domain
        );
        let resp = self.client.get(&url).send().await?;
        check_status(&resp)?;
        let body: DatasetInfoHsds = resp.json().await?;
        Ok(body)
    }

    /// Read all values from a dataset as `f64`.
    ///
    /// This fetches the JSON value response from the HSDS server and
    /// converts all numeric values to `f64`. Works for integer and float
    /// datasets. Flattens multi-dimensional arrays.
    pub async fn read_dataset_values(
        &self,
        domain: &str,
        dataset_id: &str,
    ) -> Result<Vec<f64>, HsdsError> {
        let url = format!(
            "{}/datasets/{}/value?host={}",
            self.base_url, dataset_id, domain
        );
        let resp = self.client.get(&url).send().await?;
        check_status(&resp)?;
        let body: ValueResponse = resp.json().await?;
        flatten_to_f64(&body.value)
    }

    /// Returns the base URL this client is configured for.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Check the HTTP status code and return an error for non-success.
fn check_status(resp: &reqwest::Response) -> Result<(), HsdsError> {
    let status = resp.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(HsdsError::Server {
            status: status.as_u16(),
            message: status.canonical_reason().unwrap_or("unknown").to_string(),
        })
    }
}

/// Recursively flatten a JSON value (potentially nested arrays) to f64 values.
fn flatten_to_f64(value: &serde_json::Value) -> Result<Vec<f64>, HsdsError> {
    let mut result = Vec::new();
    flatten_recursive(value, &mut result)?;
    Ok(result)
}

fn flatten_recursive(value: &serde_json::Value, out: &mut Vec<f64>) -> Result<(), HsdsError> {
    match value {
        serde_json::Value::Number(n) => {
            let v = n
                .as_f64()
                .ok_or_else(|| HsdsError::DataError("cannot convert number to f64".into()))?;
            out.push(v);
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                flatten_recursive(item, out)?;
            }
        }
        _ => {
            return Err(HsdsError::DataError(format!(
                "unexpected JSON value type: expected number or array, got {}",
                value_type_name(value)
            )));
        }
    }
    Ok(())
}

fn value_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Map an HSDS type info to a simplified type description.
pub fn map_hsds_type(type_info: &TypeInfo) -> String {
    match type_info.class.as_str() {
        "H5T_FLOAT" => {
            if let Some(base) = &type_info.base {
                match base.as_str() {
                    "H5T_IEEE_F32LE" | "H5T_IEEE_F32BE" => "f32".to_string(),
                    "H5T_IEEE_F64LE" | "H5T_IEEE_F64BE" => "f64".to_string(),
                    other => format!("float({other})"),
                }
            } else {
                "f64".to_string()
            }
        }
        "H5T_INTEGER" => {
            if let Some(base) = &type_info.base {
                match base.as_str() {
                    "H5T_STD_I8LE" | "H5T_STD_I8BE" => "i8".to_string(),
                    "H5T_STD_I16LE" | "H5T_STD_I16BE" => "i16".to_string(),
                    "H5T_STD_I32LE" | "H5T_STD_I32BE" => "i32".to_string(),
                    "H5T_STD_I64LE" | "H5T_STD_I64BE" => "i64".to_string(),
                    "H5T_STD_U8LE" | "H5T_STD_U8BE" => "u8".to_string(),
                    "H5T_STD_U16LE" | "H5T_STD_U16BE" => "u16".to_string(),
                    "H5T_STD_U32LE" | "H5T_STD_U32BE" => "u32".to_string(),
                    "H5T_STD_U64LE" | "H5T_STD_U64BE" => "u64".to_string(),
                    other => format!("int({other})"),
                }
            } else {
                "i64".to_string()
            }
        }
        "H5T_STRING" => "string".to_string(),
        other => other.to_string(),
    }
}

/// Map an HSDS shape to a dimensions vector.
pub fn map_hsds_shape(shape_info: &ShapeInfo) -> Vec<u64> {
    match shape_info.class.as_str() {
        "H5S_SIMPLE" => shape_info.dims.clone(),
        "H5S_SCALAR" => vec![],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Unit tests for flatten_to_f64 ---

    #[test]
    fn flatten_single_number() {
        let val = serde_json::json!(42.0);
        let result = flatten_to_f64(&val).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn flatten_1d_array() {
        let val = serde_json::json!([1.0, 2.0, 3.0]);
        let result = flatten_to_f64(&val).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flatten_2d_array() {
        let val = serde_json::json!([[1.0, 2.0], [3.0, 4.0]]);
        let result = flatten_to_f64(&val).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flatten_integer_values() {
        let val = serde_json::json!([1, 2, 3]);
        let result = flatten_to_f64(&val).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flatten_mixed_nested() {
        let val = serde_json::json!([[[1.0]], [[2.0]], [[3.0]]]);
        let result = flatten_to_f64(&val).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flatten_empty_array() {
        let val = serde_json::json!([]);
        let result = flatten_to_f64(&val).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn flatten_string_error() {
        let val = serde_json::json!("not a number");
        let result = flatten_to_f64(&val);
        assert!(result.is_err());
    }

    // --- map_hsds_type tests ---

    #[test]
    fn map_type_f64() {
        let info = TypeInfo {
            class: "H5T_FLOAT".to_string(),
            base: Some("H5T_IEEE_F64LE".to_string()),
        };
        assert_eq!(map_hsds_type(&info), "f64");
    }

    #[test]
    fn map_type_f32() {
        let info = TypeInfo {
            class: "H5T_FLOAT".to_string(),
            base: Some("H5T_IEEE_F32LE".to_string()),
        };
        assert_eq!(map_hsds_type(&info), "f32");
    }

    #[test]
    fn map_type_i32() {
        let info = TypeInfo {
            class: "H5T_INTEGER".to_string(),
            base: Some("H5T_STD_I32LE".to_string()),
        };
        assert_eq!(map_hsds_type(&info), "i32");
    }

    #[test]
    fn map_type_u64() {
        let info = TypeInfo {
            class: "H5T_INTEGER".to_string(),
            base: Some("H5T_STD_U64LE".to_string()),
        };
        assert_eq!(map_hsds_type(&info), "u64");
    }

    #[test]
    fn map_type_string() {
        let info = TypeInfo {
            class: "H5T_STRING".to_string(),
            base: None,
        };
        assert_eq!(map_hsds_type(&info), "string");
    }

    #[test]
    fn map_type_unknown() {
        let info = TypeInfo {
            class: "H5T_COMPOUND".to_string(),
            base: None,
        };
        assert_eq!(map_hsds_type(&info), "H5T_COMPOUND");
    }

    // --- map_hsds_shape tests ---

    #[test]
    fn map_shape_simple() {
        let shape = ShapeInfo {
            class: "H5S_SIMPLE".to_string(),
            dims: vec![3, 4],
            maxdims: vec![3, 4],
        };
        assert_eq!(map_hsds_shape(&shape), vec![3, 4]);
    }

    #[test]
    fn map_shape_scalar() {
        let shape = ShapeInfo {
            class: "H5S_SCALAR".to_string(),
            dims: vec![],
            maxdims: vec![],
        };
        assert_eq!(map_hsds_shape(&shape), Vec::<u64>::new());
    }

    // --- DomainInfo deserialization ---

    #[test]
    fn deserialize_domain_info() {
        let json = r#"{
            "root": "g-abc123",
            "created": 1700000000.0,
            "lastModified": 1700001000.0,
            "owner": "admin"
        }"#;
        let info: DomainInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.root, "g-abc123");
        assert_eq!(info.owner, "admin");
    }

    // --- DatasetInfoHsds deserialization ---

    #[test]
    fn deserialize_dataset_info() {
        let json = r#"{
            "id": "d-xyz789",
            "shape": {
                "class": "H5S_SIMPLE",
                "dims": [10, 20],
                "maxdims": [10, 20]
            },
            "type": {
                "class": "H5T_FLOAT",
                "base": "H5T_IEEE_F64LE"
            }
        }"#;
        let info: DatasetInfoHsds = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "d-xyz789");
        assert_eq!(info.shape.dims, vec![10, 20]);
        assert_eq!(info.dtype.class, "H5T_FLOAT");
    }

    // --- LinkInfo deserialization ---

    #[test]
    fn deserialize_link_info() {
        let json = r#"{
            "title": "dataset1",
            "class": "H5L_TYPE_HARD",
            "id": "d-abc",
            "target": {
                "id": "d-abc",
                "collection": "datasets"
            }
        }"#;
        let info: LinkInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.title, "dataset1");
        assert_eq!(info.class, "H5L_TYPE_HARD");
        assert_eq!(info.id.as_deref(), Some("d-abc"));
    }

    // --- HsdsClient construction ---

    #[test]
    fn client_base_url_trailing_slash() {
        let client = HsdsClient::new("http://localhost:5101/");
        assert_eq!(client.base_url(), "http://localhost:5101");
    }

    #[test]
    fn client_base_url_no_trailing_slash() {
        let client = HsdsClient::new("http://localhost:5101");
        assert_eq!(client.base_url(), "http://localhost:5101");
    }

    // --- HsdsError display ---

    #[test]
    fn error_display_server() {
        let err = HsdsError::Server {
            status: 404,
            message: "Not Found".to_string(),
        };
        assert!(err.to_string().contains("404"));
        assert!(err.to_string().contains("Not Found"));
    }

    #[test]
    fn error_display_data() {
        let err = HsdsError::DataError("bad data".to_string());
        assert!(err.to_string().contains("bad data"));
    }

    // --- ValueResponse deserialization ---

    #[test]
    fn deserialize_value_response_1d() {
        let json = r#"{"value": [1.0, 2.0, 3.0]}"#;
        let resp: ValueResponse = serde_json::from_str(json).unwrap();
        let values = flatten_to_f64(&resp.value).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn deserialize_value_response_2d() {
        let json = r#"{"value": [[1, 2], [3, 4]]}"#;
        let resp: ValueResponse = serde_json::from_str(json).unwrap();
        let values = flatten_to_f64(&resp.value).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn map_integer_types_exhaustive() {
        let cases = [
            ("H5T_STD_I8LE", "i8"),
            ("H5T_STD_I16BE", "i16"),
            ("H5T_STD_I64LE", "i64"),
            ("H5T_STD_U8BE", "u8"),
            ("H5T_STD_U16LE", "u16"),
            ("H5T_STD_U32BE", "u32"),
        ];
        for (base, expected) in &cases {
            let info = TypeInfo {
                class: "H5T_INTEGER".to_string(),
                base: Some(base.to_string()),
            };
            assert_eq!(map_hsds_type(&info), *expected);
        }
    }

    #[test]
    fn map_float_default_no_base() {
        let info = TypeInfo {
            class: "H5T_FLOAT".to_string(),
            base: None,
        };
        assert_eq!(map_hsds_type(&info), "f64");
    }

    #[test]
    fn map_integer_default_no_base() {
        let info = TypeInfo {
            class: "H5T_INTEGER".to_string(),
            base: None,
        };
        assert_eq!(map_hsds_type(&info), "i64");
    }
}
