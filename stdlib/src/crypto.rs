/*!
 * Cryptographic utilities for SYNTH
 */

use sha2::{Sha256, Sha512, Digest};
use blake3;

/// SHA-256 hash
pub fn sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// SHA-512 hash
pub fn sha512(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha512::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// BLAKE3 hash
pub fn blake3_hash(data: &[u8]) -> Vec<u8> {
    blake3::hash(data).as_bytes().to_vec()
}

/// Convert bytes to hex string
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Convert hex string to bytes
pub fn from_hex(hex: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(hex)
}