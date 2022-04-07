use anyhow::Result;
use log::debug;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};

mod decode;
use decode::product_sum;

type Alphabet = i64;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn bp_decode<'py>(
    py: Python<'py>,
    parity_check: PyReadonlyArray2<'_, Alphabet>,
    error_rate: PyReadonlyArray1<f64>,
    max_iter: usize,
    syndrome: PyReadonlyArray1<Alphabet>,
) -> Result<&'py PyArray1<Alphabet>> {
    debug!("Entered rust's side of implementation!");
    let parity_check = parity_check.as_array();
    let error_rate = error_rate.as_array();
    let syndrome = syndrome.as_array();
    let decoded = product_sum(parity_check, error_rate, max_iter, syndrome)?;

    Ok(decoded.into_pyarray(py))
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
