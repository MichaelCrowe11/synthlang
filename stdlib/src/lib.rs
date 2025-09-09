/*!
 * SYNTH Standard Library
 * Core functions and utilities for SYNTH programs
 */

#![warn(missing_docs)]
#![allow(clippy::module_inception)]

// Core modules
pub mod core;
pub mod string;
pub mod math;
pub mod collections;
pub mod io;
pub mod time;
pub mod random;

// Advanced modules
pub mod crypto;
pub mod net;
pub mod json;
pub mod regex;

// AI-specific modules
#[cfg(feature = "ai")]
pub mod ai;

// Quantum-specific modules  
#[cfg(feature = "quantum")]
pub mod quantum;

// Re-export commonly used types and functions
pub use crate::core::*;
pub use crate::string::*;
pub use crate::math::*;
pub use crate::collections::*;

/// Version information for the standard library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the SYNTH standard library
/// This should be called at the beginning of SYNTH program execution
pub fn init() -> anyhow::Result<()> {
    // Initialize logging
    #[cfg(feature = "std")]
    {
        if std::env::var("SYNTH_LOG").is_ok() {
            env_logger::init();
        }
    }

    // Initialize AI subsystem if available
    #[cfg(feature = "ai")]
    {
        ai::init()?;
    }

    // Initialize quantum subsystem if available
    #[cfg(feature = "quantum")]
    {
        quantum::init()?;
    }

    Ok(())
}

/// Global configuration for the standard library
#[derive(Debug, Clone)]
pub struct Config {
    /// Enable debug output
    pub debug: bool,
    /// AI provider configuration
    #[cfg(feature = "ai")]
    pub ai_config: Option<ai::Config>,
    /// Quantum backend configuration
    #[cfg(feature = "quantum")]
    pub quantum_config: Option<quantum::Config>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            debug: false,
            #[cfg(feature = "ai")]
            ai_config: None,
            #[cfg(feature = "quantum")]
            quantum_config: None,
        }
    }
}

/// Runtime context for SYNTH programs
pub struct Runtime {
    config: Config,
    start_time: std::time::Instant,
}

impl Runtime {
    /// Create a new runtime with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Create a new runtime with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            start_time: std::time::Instant::now(),
        }
    }

    /// Get the runtime configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the runtime uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get memory usage information
    pub fn memory_usage(&self) -> MemoryInfo {
        // This is a simplified implementation
        // In a real implementation, we'd use platform-specific APIs
        MemoryInfo {
            heap_used: 0,
            heap_total: 0,
            stack_used: 0,
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage information
#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    /// Heap memory used (bytes)
    pub heap_used: usize,
    /// Total heap memory (bytes)
    pub heap_total: usize,
    /// Stack memory used (bytes)
    pub stack_used: usize,
}

/// Error types for the standard library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// Regex error
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
    
    /// AI error
    #[cfg(feature = "ai")]
    #[error("AI error: {0}")]
    Ai(#[from] ai::Error),
    
    /// Quantum error
    #[cfg(feature = "quantum")]
    #[error("Quantum error: {0}")]
    Quantum(#[from] quantum::Error),
    
    /// Generic error
    #[error("Error: {0}")]
    Generic(String),
}