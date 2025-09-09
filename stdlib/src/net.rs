/*!
 * Network utilities for SYNTH
 */

use std::net::{IpAddr, SocketAddr};

/// Parse IP address
pub fn parse_ip(s: &str) -> Result<IpAddr, std::net::AddrParseError> {
    s.parse()
}

/// Parse socket address
pub fn parse_socket(s: &str) -> Result<SocketAddr, std::net::AddrParseError> {
    s.parse()
}

/// Check if IP is IPv4
pub fn is_ipv4(addr: &IpAddr) -> bool {
    addr.is_ipv4()
}

/// Check if IP is IPv6
pub fn is_ipv6(addr: &IpAddr) -> bool {
    addr.is_ipv6()
}