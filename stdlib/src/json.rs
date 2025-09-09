/*!
 * JSON utilities for SYNTH
 */

use serde_json::{Value, Map};

/// Parse JSON string
pub fn parse(s: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(s)
}

/// Stringify JSON value
pub fn stringify(value: &Value) -> String {
    value.to_string()
}

/// Pretty print JSON
pub fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Create JSON object
pub fn object() -> Map<String, Value> {
    Map::new()
}

/// Create JSON array
pub fn array() -> Vec<Value> {
    Vec::new()
}