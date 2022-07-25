// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use anyhow::Result;
use log::debug;
use numpy::PyReadonlyArray2;
use pyo3::{pyclass, pymethods, pymodule, types::PyModule, PyResult, Python};

mod decoder;
pub use decoder::{Decoder, FloatType};
mod decoder_special;
pub use decoder_special::{DecoderSpecial};


macro_rules! register_py_decoder_special_class {
    ($m:ident <= $Name:ident{N: $N:literal, R: $R:literal, DV: $DV:literal, DC: $DC:literal, B: $B:literal, BSUM: $BSUM:literal}) => {{
        type CustomDecoder = DecoderSpecial<$N, $R, {$N-$R}, {$DC-1}, $DC, $DV, $B, {$B * 2 + 1}, $BSUM, {$BSUM * 2 + 1}, i8>;

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
                    "Constructing decoder {} with N={N}, R={R}, DV={DV}, DC={DC}, BSUM={BSUM}, Input parity check matrix has the shape: {shape:?}",
                    stringify!($Name),
                    N = $N,
                    R = $R,
                    DV = $DV,
                    DC = $DC,
                    BSUM = $BSUM,
                    shape = py_parity_check.shape()
                );
                let mut parity_check = [[0; $N]; $R];
                for row in 0..parity_check.len() {
                    for col in 0..parity_check[row].len() {
                        parity_check[row][col] = py_parity_check[(row, col)];
                    }
                }
                Ok($Name {
                    decoder: DecoderSpecial::new(parity_check, iterations),
                })
            }

            /// min_sum algorithm
            ///
            /// This method is parallelizable from python.
            ///
            /// Attempts have  been made to make this function parallel from within,
            /// but that resulted in performance loss
            fn min_sum(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_sum: PyReadonlyArray2<FloatType>) -> Result<[i8; $N]> {
                let py_channel_output = py_channel_output.as_array();
                let py_channel_output_sum = py_channel_output_sum.as_array();
                py.allow_threads(||{
                    let mut channel_output = [[0.0; {$B * 2 + 1}]; {$N-$R}];
                    let mut channel_output_sum = [[0.0; {$BSUM * 2 + 1}]; $R];
                    for variable in 0..channel_output.len() {
                        for value in 0..channel_output[variable].len() {
                            channel_output[variable][value] = py_channel_output[(variable, value)].into();
                        }
                    }
                    for variable in 0..channel_output_sum.len() {
                        for value in 0..channel_output_sum[variable].len() {
                            channel_output_sum[variable][value] = py_channel_output_sum[(variable, value)].into();
                        }
                    }
                    let channel_llr = CustomDecoder::into_llr(&channel_output);
                    let channel_llr_sum = CustomDecoder::into_llr(&channel_output_sum);
                    self.decoder.min_sum(channel_llr, channel_llr_sum)
                })
            }
        }

        $m.add_class::<$Name>()?;
    }};
}

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
        type CustomDecoder = Decoder<$N, $R, $DC, $DV, {$B * 2 + 1}, $B, i8>;

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
        m <= DecoderN6R3V3C4GF16 {
            N: 6,
            R: 3,
            DV: 3,
            DC: 4,
            B: 7
        }
    );

    // A slightly larger decoder for testing
    register_py_decoder_class!(
        m <= DecoderN450R150V3C7GF16 {
            N: 450,
            R: 150,
            DV: 3,
            DC: 7,
            B: 7
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

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
