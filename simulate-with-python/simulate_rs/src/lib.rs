use log::debug;
use pyo3::prelude::*;

mod decode;
use decode::bp_decode_impl;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn bp_decode(a: usize, b: usize) -> PyResult<String> {
    debug!("Entered rust's side of implementation!");
    Ok(bp_decode_impl(a, b))
}

/// A Python module implemented in Rust.
#[pymodule]
fn simulate_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(bp_decode, m)?)?;
    Ok(())
}
