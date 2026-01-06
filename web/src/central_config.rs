//! Centralized configuration loading from config.toml.
//!
//! This module provides a single source of truth for configuration values,
//! loaded from config.toml at the project root with support for environment
//! variable overrides using the CARTRIDGE_* naming convention.
//!
//! Environment Variable Override Pattern:
//!     CARTRIDGE_<SECTION>_<KEY>=value
//!
//!     Examples:
//!         CARTRIDGE_COMMON_ENV_ID=connect4
//!         CARTRIDGE_COMMON_DATA_DIR=/data
//!         CARTRIDGE_WEB_HOST=127.0.0.1
//!         CARTRIDGE_WEB_PORT=3000

use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, info, warn};

// Default constants matching actor/trainer
mod defaults {
    pub const DATA_DIR: &str = "./data";
    pub const ENV_ID: &str = "tictactoe";
    pub const LOG_LEVEL: &str = "info";
    pub const HOST: &str = "0.0.0.0";
    pub const PORT: u16 = 8080;
    pub const MODEL_BACKEND: &str = "filesystem";
    pub const POSTGRES_URL: &str = "postgresql://cartridge:cartridge@localhost:5432/cartridge";
}

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default)]
pub struct CentralConfig {
    #[serde(default)]
    pub common: CommonConfig,
    #[serde(default)]
    pub web: WebConfig,
    #[serde(default)]
    pub storage: StorageConfig,
}

// Serde default functions
fn d_data_dir() -> String {
    defaults::DATA_DIR.into()
}
fn d_env_id() -> String {
    defaults::ENV_ID.into()
}
fn d_log_level() -> String {
    defaults::LOG_LEVEL.into()
}
fn d_host() -> String {
    defaults::HOST.into()
}
fn d_port() -> u16 {
    defaults::PORT
}
fn d_model_backend() -> String {
    defaults::MODEL_BACKEND.into()
}
fn d_postgres_url() -> Option<String> {
    Some(defaults::POSTGRES_URL.into())
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CommonConfig {
    #[serde(default = "d_data_dir")]
    pub data_dir: String,
    #[serde(default = "d_env_id")]
    pub env_id: String,
    #[serde(default = "d_log_level")]
    pub log_level: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            data_dir: defaults::DATA_DIR.into(),
            env_id: defaults::ENV_ID.into(),
            log_level: defaults::LOG_LEVEL.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    #[serde(default = "d_host")]
    pub host: String,
    #[serde(default = "d_port")]
    pub port: u16,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: defaults::HOST.into(),
            port: defaults::PORT,
        }
    }
}

/// Storage backend configuration
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    #[serde(default = "d_model_backend")]
    pub model_backend: String,
    #[serde(default = "d_postgres_url")]
    pub postgres_url: Option<String>,
    #[serde(default)]
    pub s3_bucket: Option<String>,
    #[serde(default)]
    pub s3_endpoint: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            model_backend: defaults::MODEL_BACKEND.into(),
            postgres_url: Some(defaults::POSTGRES_URL.into()),
            s3_bucket: None,
            s3_endpoint: None,
        }
    }
}

/// Standard locations to search for config.toml
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",      // Current directory
    "../config.toml",   // Parent directory (when running from web/)
    "/app/config.toml", // Docker container
];

/// Load the central configuration from config.toml.
pub fn load_config() -> CentralConfig {
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
    apply_env_overrides(CentralConfig::default())
}

fn load_from_path(path: &PathBuf) -> CentralConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => apply_env_overrides(config),
            Err(e) => {
                warn!("Failed to parse {}: {}, using defaults", path.display(), e);
                apply_env_overrides(CentralConfig::default())
            }
        },
        Err(e) => {
            warn!("Failed to read {}: {}, using defaults", path.display(), e);
            apply_env_overrides(CentralConfig::default())
        }
    }
}

/// Macro to reduce env override boilerplate
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = v;
        }
    };
    // Parseable field (u16, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Ok(v) =
            std::env::var($key).and_then(|s| s.parse().map_err(|_| std::env::VarError::NotPresent))
        {
            $config.$section.$field = v;
        }
    };
    // Optional string field
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = Some(v);
        }
    };
}

fn apply_env_overrides(mut config: CentralConfig) -> CentralConfig {
    // Common section
    env_override!(config, common.data_dir, "CARTRIDGE_COMMON_DATA_DIR");
    env_override!(config, common.env_id, "CARTRIDGE_COMMON_ENV_ID");
    env_override!(config, common.log_level, "CARTRIDGE_COMMON_LOG_LEVEL");

    // Web section
    env_override!(config, web.host, "CARTRIDGE_WEB_HOST");
    env_override!(config, web.port, "CARTRIDGE_WEB_PORT", parse);

    // Storage section
    env_override!(
        config,
        storage.model_backend,
        "CARTRIDGE_STORAGE_MODEL_BACKEND"
    );
    env_override!(
        config,
        storage.postgres_url,
        "CARTRIDGE_STORAGE_POSTGRES_URL",
        optional
    );
    env_override!(
        config,
        storage.s3_bucket,
        "CARTRIDGE_STORAGE_S3_BUCKET",
        optional
    );
    env_override!(
        config,
        storage.s3_endpoint,
        "CARTRIDGE_STORAGE_S3_ENDPOINT",
        optional
    );

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CentralConfig::default();
        assert_eq!(config.common.env_id, "tictactoe");
        assert_eq!(config.common.data_dir, "./data");
        assert_eq!(config.web.host, "0.0.0.0");
        assert_eq!(config.web.port, 8080);
    }

    #[test]
    fn test_cartridge_env_overrides() {
        std::env::set_var("CARTRIDGE_COMMON_ENV_ID", "connect4");
        std::env::set_var("CARTRIDGE_COMMON_DATA_DIR", "/custom/data");
        std::env::set_var("CARTRIDGE_WEB_PORT", "3000");

        let config = load_config();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "/custom/data");
        assert_eq!(config.web.port, 3000);

        std::env::remove_var("CARTRIDGE_COMMON_ENV_ID");
        std::env::remove_var("CARTRIDGE_COMMON_DATA_DIR");
        std::env::remove_var("CARTRIDGE_WEB_PORT");
    }

    #[test]
    fn test_parse_config_toml() {
        let toml_content = r#"
[common]
env_id = "connect4"
data_dir = "/custom/data"

[web]
host = "127.0.0.1"
port = 3000
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "/custom/data");
        assert_eq!(config.web.host, "127.0.0.1");
        assert_eq!(config.web.port, 3000);
    }

    #[test]
    fn test_partial_config() {
        let toml_content = r#"
[common]
env_id = "connect4"
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "./data"); // Default
        assert_eq!(config.web.port, 8080); // Default
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = CentralConfig::default();
        assert_eq!(config.storage.model_backend, "filesystem");
        assert_eq!(
            config.storage.postgres_url,
            Some("postgresql://cartridge:cartridge@localhost:5432/cartridge".to_string())
        );
        assert!(config.storage.s3_bucket.is_none());
    }

    #[test]
    fn test_storage_env_overrides() {
        std::env::set_var("CARTRIDGE_STORAGE_MODEL_BACKEND", "s3");
        std::env::set_var("CARTRIDGE_STORAGE_S3_BUCKET", "my-bucket");

        let config = load_config();
        assert_eq!(config.storage.model_backend, "s3");
        assert_eq!(config.storage.s3_bucket, Some("my-bucket".to_string()));

        std::env::remove_var("CARTRIDGE_STORAGE_MODEL_BACKEND");
        std::env::remove_var("CARTRIDGE_STORAGE_S3_BUCKET");
    }
}
