//! Dictionary encoding for string datasets.
//!
//! Dictionary encoding replaces repeated strings with integer indices into
//! a shared dictionary. This is effective for high-repetition fields like
//! tags, source channels, and categories.
//!
//! # Example
//!
//! ```rust
//! use rustyhdf5_format::dict_encoding::DictionaryEncoder;
//!
//! let strings = vec!["cat", "dog", "cat", "bird", "dog", "cat"];
//! let encoded = DictionaryEncoder::encode(&strings);
//! assert_eq!(encoded.indices, vec![0, 1, 0, 2, 1, 0]);
//! assert_eq!(encoded.dictionary, vec!["cat", "dog", "bird"]);
//!
//! // Decode back
//! let decoded = encoded.decode();
//! assert_eq!(decoded, strings);
//! ```

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;

/// Result of dictionary encoding a string array.
#[derive(Debug, Clone, PartialEq)]
pub struct DictEncoded {
    /// Unique strings in order of first appearance.
    pub dictionary: Vec<String>,
    /// Index into `dictionary` for each original string element.
    pub indices: Vec<i32>,
}

/// Dictionary encoder for string datasets.
pub struct DictionaryEncoder;

impl DictionaryEncoder {
    /// Encode a slice of strings into a dictionary + index array.
    ///
    /// Strings are assigned indices in order of first appearance.
    /// Returns a [`DictEncoded`] with the unique dictionary and the
    /// per-element index array.
    pub fn encode(strings: &[&str]) -> DictEncoded {
        #[cfg(feature = "std")]
        let mut map: HashMap<&str, i32> = HashMap::new();
        #[cfg(not(feature = "std"))]
        let mut map: BTreeMap<&str, i32> = BTreeMap::new();

        let mut dictionary = Vec::new();
        let mut indices = Vec::with_capacity(strings.len());

        for &s in strings {
            let idx = if let Some(&existing) = map.get(s) {
                existing
            } else {
                let idx = dictionary.len() as i32;
                dictionary.push(String::from(s));
                map.insert(s, idx);
                idx
            };
            indices.push(idx);
        }

        DictEncoded {
            dictionary,
            indices,
        }
    }

    /// Encode a slice of owned strings.
    pub fn encode_owned(strings: &[String]) -> DictEncoded {
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        Self::encode(&refs)
    }

    /// Compute the compression ratio of dictionary encoding.
    ///
    /// Returns the ratio of original total bytes to encoded bytes
    /// (dictionary strings + i32 indices). Values > 1.0 indicate
    /// space savings from encoding.
    pub fn compression_ratio(strings: &[&str]) -> f64 {
        if strings.is_empty() {
            return 1.0;
        }
        let original_bytes: usize = strings.iter().map(|s| s.len()).sum();
        let encoded = Self::encode(strings);
        let dict_bytes: usize = encoded.dictionary.iter().map(|s| s.len()).sum();
        let index_bytes = encoded.indices.len() * 4; // i32
        let encoded_bytes = dict_bytes + index_bytes;
        if encoded_bytes == 0 {
            return 1.0;
        }
        original_bytes as f64 / encoded_bytes as f64
    }

    /// Check if dictionary encoding would be beneficial for the given strings.
    ///
    /// Returns `true` if the encoding would save space (compression ratio > 1.0)
    /// and there are fewer unique values than 75% of the total count.
    pub fn is_beneficial(strings: &[&str]) -> bool {
        if strings.len() < 4 {
            return false;
        }
        let encoded = Self::encode(strings);
        let unique_ratio = encoded.dictionary.len() as f64 / strings.len() as f64;
        if unique_ratio > 0.75 {
            return false;
        }
        Self::compression_ratio(strings) > 1.0
    }
}

impl DictEncoded {
    /// Decode the dictionary-encoded data back to strings.
    pub fn decode(&self) -> Vec<&str> {
        self.indices
            .iter()
            .map(|&idx| self.dictionary[idx as usize].as_str())
            .collect()
    }

    /// Decode to owned strings.
    pub fn decode_owned(&self) -> Vec<String> {
        self.indices
            .iter()
            .map(|&idx| self.dictionary[idx as usize].clone())
            .collect()
    }

    /// Number of unique strings in the dictionary.
    pub fn unique_count(&self) -> usize {
        self.dictionary.len()
    }

    /// Number of total elements.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_basic() {
        let strings = vec!["cat", "dog", "cat", "bird", "dog", "cat"];
        let encoded = DictionaryEncoder::encode(&strings);
        assert_eq!(encoded.dictionary, vec!["cat", "dog", "bird"]);
        assert_eq!(encoded.indices, vec![0, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn roundtrip() {
        let strings = vec!["alpha", "beta", "gamma", "alpha", "beta"];
        let encoded = DictionaryEncoder::encode(&strings);
        let decoded = encoded.decode();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn encode_all_unique() {
        let strings = vec!["a", "b", "c"];
        let encoded = DictionaryEncoder::encode(&strings);
        assert_eq!(encoded.unique_count(), 3);
        assert_eq!(encoded.indices, vec![0, 1, 2]);
    }

    #[test]
    fn encode_all_same() {
        let strings = vec!["x", "x", "x", "x"];
        let encoded = DictionaryEncoder::encode(&strings);
        assert_eq!(encoded.unique_count(), 1);
        assert_eq!(encoded.indices, vec![0, 0, 0, 0]);
    }

    #[test]
    fn encode_empty() {
        let strings: Vec<&str> = vec![];
        let encoded = DictionaryEncoder::encode(&strings);
        assert!(encoded.is_empty());
        assert_eq!(encoded.unique_count(), 0);
    }

    #[test]
    fn compression_ratio_high_repetition() {
        // 100 elements but only 3 unique strings
        let mut strings = Vec::new();
        for i in 0..100 {
            strings.push(match i % 3 {
                0 => "category_alpha",
                1 => "category_beta",
                _ => "category_gamma",
            });
        }
        let ratio = DictionaryEncoder::compression_ratio(&strings);
        assert!(ratio > 1.0, "ratio={ratio} should be > 1.0");
    }

    #[test]
    fn is_beneficial_high_repetition() {
        let strings: Vec<&str> = (0..100)
            .map(|i| match i % 3 {
                0 => "category_alpha",
                1 => "category_beta",
                _ => "category_gamma",
            })
            .collect();
        assert!(DictionaryEncoder::is_beneficial(&strings));
    }

    #[test]
    fn is_beneficial_all_unique() {
        let owned: Vec<String> = (0..100).map(|i| format!("unique_{i}")).collect();
        let strings: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        assert!(!DictionaryEncoder::is_beneficial(&strings));
    }

    #[test]
    fn encode_owned_strings() {
        let strings = vec![String::from("hello"), String::from("world"), String::from("hello")];
        let encoded = DictionaryEncoder::encode_owned(&strings);
        assert_eq!(encoded.indices, vec![0, 1, 0]);
        assert_eq!(encoded.dictionary, vec!["hello", "world"]);
    }

    #[test]
    fn decode_owned() {
        let strings = vec!["x", "y", "x"];
        let encoded = DictionaryEncoder::encode(&strings);
        let decoded = encoded.decode_owned();
        assert_eq!(decoded, vec!["x", "y", "x"]);
    }
}
