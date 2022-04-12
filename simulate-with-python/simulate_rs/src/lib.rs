// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use anyhow::Result;
use log::debug;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};

mod decode;

type Alphabet = u8;

g2p::g2p!(GF16, 4, modulus: 0b10011);

/// Formats the sum of two numbers as string.
#[pyfunction]
fn bp_decode<'py>(
    _py: Python<'py>,
    _parity_check: PyReadonlyArray2<'_, Alphabet>,
    _error_rate: PyReadonlyArray1<f64>,
    _max_iter: usize,
    _syndrome: PyReadonlyArray1<Alphabet>,
) -> Result<&'py PyArray1<Alphabet>> {
    debug!("Entered rust's side of implementation!");

    todo!();
}

/// A Python module implemented in Rust.
#[pymodule]
fn simulate_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(bp_decode, m)?)?;
    debug!("Rust package simulate_rs imported!");
    Ok(())
}
