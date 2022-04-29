// This is for the g2p macro
#![allow(clippy::suspicious_arithmetic_impl)]
use anyhow::Result;
use g2p::GaloisField;
use log::debug;
use numpy::PyReadonlyArray2;
use pyo3::{pyclass, pymethods, pymodule, types::PyModule, PyResult, Python};

mod decoder;
pub use decoder::Decoder;

g2p::g2p!(GF16, 4, modulus: 0b10011);

macro_rules! register_py_decoder_class {
    ($m:ident <= $Name:ident{N: $N:literal, R: $R:literal, DV: $DV:literal, DC: $DC:literal, GF: $GF:ident}) => {{
        type CustomDecoder = Decoder<$N, $R, $DV, $DC, { $GF::SIZE }, $GF>;

        #[pyclass]
        struct $Name {
            decoder: CustomDecoder,
        }

        #[pymethods]
        impl $Name {
            #[new]
            fn new(py_parity_check: PyReadonlyArray2<u8>) -> Result<Self> {
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
                let mut parity_check = [[GF16::ZERO; $N]; $R];
                for row in 0..parity_check.len() {
                    for col in 0..parity_check[row].len() {
                        parity_check[row][col] = py_parity_check[(row, col)].into();
                    }
                }
                Ok($Name {
                    decoder: Decoder::new(parity_check, 10),
                })
            }

            fn min_sum(&self, py_channel_output: PyReadonlyArray2<f64>) -> Result<Vec<u8>> {
                let py_channel_output = py_channel_output.as_array();
                let mut channel_output = [[0.0; { $GF::SIZE }]; $N];
                for variable in 0..channel_output.len() {
                    for value in 0..channel_output[variable].len() {
                        channel_output[variable][value] = py_channel_output[(variable, value)].into();
                    }
                }
                let channel_llr = CustomDecoder::into_llr(&channel_output);
                let l = self.decoder.min_sum(channel_llr)?;
                let res: Vec<u8> = l.iter().copied().map(GF16::into).collect();
                Ok(res)
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

    register_py_decoder_class!(
        m <= DecoderN6R3V4C3GF16 {
            N: 6,
            R: 3,
            DV: 4,
            DC: 3,
            GF: GF16
        }
    );

    register_py_decoder_class!(
        m <= DecoderN450R150V7C3GF16 {
            N: 450,
            R: 150,
            DV: 7,
            DC: 3,
            GF: GF16
        }
    );

    debug!("Rust package simulate_rs imported!");
    Ok(())
}
