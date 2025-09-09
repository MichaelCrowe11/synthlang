/*!
 * Regular expression utilities for SYNTH
 */

pub use regex::{Regex, Match, Captures};

/// Create new regex
pub fn compile(pattern: &str) -> Result<Regex, regex::Error> {
    Regex::new(pattern)
}

/// Test if pattern matches
pub fn is_match(pattern: &str, text: &str) -> Result<bool, regex::Error> {
    Ok(Regex::new(pattern)?.is_match(text))
}

/// Find first match
pub fn find(pattern: &str, text: &str) -> Result<Option<String>, regex::Error> {
    Ok(Regex::new(pattern)?.find(text).map(|m| m.as_str().to_string()))
}

/// Find all matches
pub fn find_all(pattern: &str, text: &str) -> Result<Vec<String>, regex::Error> {
    Ok(Regex::new(pattern)?
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect())
}

/// Replace all matches
pub fn replace_all(pattern: &str, text: &str, replacement: &str) -> Result<String, regex::Error> {
    Ok(Regex::new(pattern)?.replace_all(text, replacement).to_string())
}