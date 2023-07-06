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
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::ParserError(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
