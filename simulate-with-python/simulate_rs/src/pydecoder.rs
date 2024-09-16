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

/// Use this macro to create new special q-ary decoders for different sizes/parameters
///
/// # Usage:
///
/// register_py_decoder_special_class!(module <= Name{
///     N: <number of columns in H, i.e. number or variable+check nodes>,
///     R: <number of rows in H, i.e. number or check nodes>,
///     DV: <Maximum variable node degree (num checks, per variable)>,
///     DC: <Maximum check node degree (num variables, per check)>,
///     B: <first N-R variables assumed to have values from the range [-B, ..., 0, ..., B]>
///     BSUM: <last R variables assumed to have values from the range [-BSUM, ..., 0, ..., BSUM]>
/// });
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
                // let mut parity_check = [[0; $N]; $R];
                // for row in 0..parity_check.len() {
                //     for col in 0..parity_check[row].len() {
                //         parity_check[row][col] = py_parity_check[(row, col)];
                //     }
                // }
                // let parity_check: Box<[Box<[i8]>]> = Box::new(
                //     (0..$R)
                //         .map(|row_idx| {
                //             let row = py_parity_check.slice(ndarray::s![row_idx, ..]);
                //             let row_vec: Vec<i8> = row.to_vec();
                //             Box::new(row_vec.into_boxed_slice())
                //         })
                //         .collect()
                // );
                let mut parity_check: Vec<Box<[i8]>> = Vec::with_capacity($R);
                
                for row in py_parity_check.outer_iter() {
                    let vec_row: Vec<i8> = row.to_vec();
                    let boxed_row: Box<[i8]> = vec_row.into_boxed_slice();
                    parity_check.push(boxed_row);
                }
                Ok($Name {
                    decoder: DecoderSpecial::new(parity_check.into_boxed_slice(), iterations),
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
