// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use anyhow::Result;
use log::debug;
use numpy::PyReadonlyArray2;
use pyo3::{pyclass, pymethods, pymodule, types::PyModule, PyResult, Python};

mod decoder;
pub use decoder::{Decoder, FloatType};

/// Use this macro to create new decoders for different sizes/parameters
///
/// # Usage:
///
/// register_py_decoder_class!(module <= Name{
///     N: <number of variable nodes>,
///     R: <number of check nodes>,
///     DV: <Maximum variable node degree (num checks, per variable)>,
///     DC: <Maximum check node degree (num variables, per check)>,
///     GF: Galois Field to operate on. ("Q" submessages)
/// });
macro_rules! register_py_decoder_class {
    ($m:ident <= $Name:ident{N: $N:literal, R: $R:literal, DV: $DV:literal, DC: $DC:literal, B: $B:literal}) => {{
        type CustomDecoder = Decoder<$N, $R, $DV, $DC, {$B * 2 + 1}, $B, i8>;

        #[pyclass]
        struct $Name {
            decoder: CustomDecoder,
        }

        #[pymethods]
        impl $Name {
            #[new]
            fn new(py_parity_check: PyReadonlyArray2<i8>, iterations: u32) -> Result<Self> {
                let py_parity_check = py_parity_check.as_array();
                ::log::info!(
                    "Constructing decoder {} with N={N}, R={R}, DV={DV}, DC={DC}, GF={GF}, Input parity check matrix has the shape: {shape:?}",
                    stringify!($Name),
                    N = $N,
                    R = $R,
                    DV = $DV,
                    DC = $DC,
                    GF = stringify!($GF),
                    shape = py_parity_check.shape()
                );
                let mut parity_check = [[0; $N]; $R];
                for row in 0..parity_check.len() {
                    for col in 0..parity_check[row].len() {
                        parity_check[row][col] = py_parity_check[(row, col)];
                    }
                }
                Ok($Name {
                    decoder: Decoder::new(parity_check, iterations),
                })
            }

            /// min_sum algorithm
            ///
            /// This method is parallelizable from python.
            ///
            /// Attempts have  been made to make this function parallel from within,
            /// but that resulted in performance loss
            fn min_sum(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>) -> Result<[i8; $N]> {
                let py_channel_output = py_channel_output.as_array();
                py.allow_threads(||{
                    let mut channel_output = [[0.0; {$B * 2 + 1}]; $N];
                    for variable in 0..channel_output.len() {
                        for value in 0..channel_output[variable].len() {
                            channel_output[variable][value] = py_channel_output[(variable, value)].into();
                        }
                    }
                    let channel_llr = CustomDecoder::into_llr(&channel_output);
                    self.decoder.min_sum(channel_llr)
                })
            }
        }

        $m.add_class::<$Name>()?;
    }};
}

/// A Python module implemented in Rust.
#[pymodule]
fn simulate_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    pyo3_log::init();

    // Create a tiny toy example
    register_py_decoder_class!(
        m <= DecoderN6R3V4C3GF16 {
            N: 6,
            R: 3,
            DV: 4,
            DC: 3,
            B: 7
        }
    );

    // A slightly larger decoder for testing
    register_py_decoder_class!(
        m <= DecoderN450R150V7C3GF16 {
            N: 450,
            R: 150,
            DV: 7,
            DC: 3,
            B: 7
        }
    );

    // Kyber first 256 coefficients
    register_py_decoder_class!(
        m <= DecoderN512R256V3C2B4 {
            N: 512,
            R: 256,
            DV: 3,
            DC: 2,
            B: 4
        }
    );

    // Full Kyber-768
    register_py_decoder_class!(
        m <= DecoderN1024R256V7C2B12 {
            N: 1024,
            R: 256,
            DV: 7,
            DC: 2,
            B: 12
        }
    );

    // Full Kyber-768
    register_py_decoder_class!(
        m <= DecoderN1024R256V4C2B6 {
            N: 1024,
            R: 256,
            DV: 4,
            DC: 2,
            B: 6
        }
    );

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
