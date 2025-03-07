use pyo3::exceptions::{PyIOError, PyValueError};
use std::fmt;
use pyo3::PyErr;

#[derive(Debug)]
pub struct InvalidFragmentFileError {
    pub filename: String,
}

impl InvalidFragmentFileError {
    pub fn new(filename: &str) -> InvalidFragmentFileError {
        InvalidFragmentFileError{
            filename: filename.to_string()
        }
    }
}

impl std::error::Error for InvalidFragmentFileError{}

impl fmt::Display for InvalidFragmentFileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: Invalid fragment file.", self.filename)
    }
}

impl From<InvalidFragmentFileError> for PyErr {
    fn from(err: InvalidFragmentFileError) -> PyErr {
        PyIOError::new_err(err.to_string())
    }
}

#[derive(Debug)]
pub struct ValueError {
    pub message: String,
}

impl ValueError {
    pub fn new(message: String) -> ValueError {
        ValueError {
           message
        }
    }
}

impl std::error::Error for ValueError{}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<ValueError> for PyErr {
    fn from(err: ValueError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

