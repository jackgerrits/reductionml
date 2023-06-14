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

pub type Result<T> = std::result::Result<T, Error>;
