// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use std::{arch::x86_64::__rdtscp};
use std::arch::x86_64::__get_cpuid_max;
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
mod decoder_special;
pub use decoder_special::{DecoderSpecial};



/// A Python module implemented in Rust.
#[pymodule]
fn simulate_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    pyo3_log::init();

    // Create a tiny toy example
    register_py_decoder_class!(
        m <= DecoderN6R3V3C4B7 {
            N: 6,
            R: 3,
            DV: 3,
            DC: 4,
            B: 7
        }
    );

    // A slightly larger decoder for testing
    register_py_decoder_class!(
        m <= DecoderN450R150V3C7B1 {
            N: 450,
            R: 150,
            DV: 3,
            DC: 7,
            B: 1
        }
    );

    register_py_decoder_special_class!(
        m <= DecoderNTRU {
            N: 8,
            R: 4,
            DV: 3,
            DC: 4,
            B: 1,
            BSUM: 3
        }
    );

    // Full Kyber-768, sum_weight = 6, check_blocks = 1
    register_py_decoder_special_class!(
        m <= DecoderN1024R256SW6 {
            N: 1024, // 768 + check_blocks*256
            R: 256,
            DV: 2,
            DC: 7,
            B: 2,
            BSUM: 12
        }
    );

    // Full Kyber-768, sum_weight = 6, check_blocks = 2
    register_py_decoder_special_class!(
        m <= DecoderN1280R512SW6 {
            N: 1280,
            R: 512,
            DV: 4,
            DC: 7,
            B: 2,
            BSUM: 12
        }
    );

    register_py_hqc_class!(m <= Hqc128);
    register_py_hqc_class!(m <= Hqc192);
    register_py_hqc_class!(m <= Hqc256);

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
