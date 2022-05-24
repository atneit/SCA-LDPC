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
            fn new(py_parity_check: PyReadonlyArray2<bool>, iterations: u32) -> Result<Self> {
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
                let mut parity_check = [[false; $N]; $R];
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
