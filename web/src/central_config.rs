//! Centralized configuration loading from config.toml.
//!
//! This module provides a single source of truth for configuration values,
//! loaded from config.toml at the project root with support for environment
//! variable overrides.

use once_cell::sync::Lazy;
use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default)]
pub struct CentralConfig {
    #[serde(default)]
    pub common: CommonConfig,
    #[serde(default)]
    pub web: WebConfig,
}

#[derive(Debug, Deserialize)]
pub struct CommonConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_env_id")]
    pub env_id: String,
    #[allow(dead_code)]
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            env_id: default_env_id(),
            log_level: default_log_level(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct WebConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

// Default value functions
fn default_data_dir() -> String {
    "./data".to_string()
}
fn default_env_id() -> String {
    "tictactoe".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}

/// Standard locations to search for config.toml
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",           // Current directory
    "../config.toml",        // Parent directory (when running from web/)
    "/app/config.toml",      // Docker container
];

/// Load the central configuration from config.toml.
fn load_config_internal() -> CentralConfig {
    // Check for explicit config path
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.exists() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
        warn!(
            "CARTRIDGE_CONFIG={} not found, searching defaults",
            path.display()
        );
    }

    // Search default locations
    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return load_from_path(&path);
        }
    }

    // Fall back to defaults
    debug!("No config.toml found, using built-in defaults");
    CentralConfig::default()
}

fn load_from_path(path: &PathBuf) -> CentralConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => config,
            Err(e) => {
                warn!("Failed to parse {}: {}, using defaults", path.display(), e);
                CentralConfig::default()
            }
        },
        Err(e) => {
            warn!("Failed to read {}: {}, using defaults", path.display(), e);
            CentralConfig::default()
        }
    }
}

// Lazy-loaded global config
static CONFIG: Lazy<CentralConfig> = Lazy::new(load_config_internal);

/// Get the data directory, checking env var first then config.toml
pub fn get_data_dir() -> String {
    std::env::var("DATA_DIR").unwrap_or_else(|_| CONFIG.common.data_dir.clone())
}

/// Get the default game, checking env var first then config.toml
pub fn get_default_game() -> String {
    std::env::var("DEFAULT_GAME").unwrap_or_else(|_| CONFIG.common.env_id.clone())
}

/// Get the web server host
pub fn get_host() -> String {
    std::env::var("WEB_HOST").unwrap_or_else(|_| CONFIG.web.host.clone())
}

/// Get the web server port
pub fn get_port() -> u16 {
    std::env::var("WEB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CONFIG.web.port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CentralConfig::default();
        assert_eq!(config.common.env_id, "tictactoe");
        assert_eq!(config.common.data_dir, "./data");
        assert_eq!(config.web.port, 8080);
    }
}
