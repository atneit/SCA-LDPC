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

    // Kyber block of 256 coefficients
    register_py_decoder_special_class!(
        m <= DecoderN512R256V2C3B4 {
            N: 512,
            R: 256,
            DV: 2,
            DC: 3,
            B: 2,
            BSUM: 4
        }
    );

    // Full Kyber-768, sum_weight = 3
    register_py_decoder_special_class!(
        m <= DecoderN1024R256SW3 {
            N: 1024,
            R: 256,
            DV: 1,
            DC: 4,
            B: 2,
            BSUM: 6
        }
    );

    // Full Kyber-768, sum_weight = 6
    register_py_decoder_special_class!(
        m <= DecoderN1024R256SW6 {
            N: 1024,
            R: 256,
            DV: 2,
            DC: 7,
            B: 2,
            BSUM: 12
        }
    );

    // Full Kyber-768, sum_weight = 9
    register_py_decoder_special_class!(
        m <= DecoderN1024R256SW9 {
            N: 1024,
            R: 256,
            DV: 3,
            DC: 10,
            B: 2,
            BSUM: 18
        }
    );

    // Full Kyber-768, sum_weight = 12
    register_py_decoder_special_class!(
        m <= DecoderN1024R256SW12 {
            N: 1024,
            R: 256,
            DV: 4,
            DC: 13,
            B: 2,
            BSUM: 24
        }
    );
    
    register_py_hqc_class!(m <= Hqc128);
    register_py_hqc_class!(m <= Hqc192);
    register_py_hqc_class!(m <= Hqc256);

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
