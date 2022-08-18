// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use anyhow::Result;
use log::debug;
use numpy::PyReadonlyArray2;
use pyo3::{
    pyclass, pymethods, pymodule,
    types::{PyByteArray, PyBytes, PyModule},
    PyResult, Python,
};

mod decoder;
pub use decoder::{Decoder, FloatType};
#[macro_use]
mod hqc;
#[macro_use]
mod pydecoder;

/// A Python module implemented in Rust.
#[pymodule]
fn simulate_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    pyo3_log::init();

    // Create a tiny toy example
    register_py_decoder_class!(
        m <= DecoderN6R3V4C3B7 {
            N: 6,
            R: 3,
            DV: 4,
            DC: 3,
            B: 7
        }
    );

    // A slightly larger decoder for testing
    register_py_decoder_class!(
        m <= DecoderN450R150V7C3B7 {
            N: 450,
            R: 150,
            DV: 7,
            DC: 3,
            B: 7
        }
    );

    register_py_hqc_class!(m <= Hqc128);
    register_py_hqc_class!(m <= Hqc192);
    register_py_hqc_class!(m <= Hqc256);

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
