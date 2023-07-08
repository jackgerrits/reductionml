use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Parser error: {0}")]
    ParserError(String),
    #[error("Invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("Invalid YAML: {0}")]
    InvalidYaml(#[from] serde_yaml::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
