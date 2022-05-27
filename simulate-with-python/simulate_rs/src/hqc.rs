macro_rules! register_py_hqc_class {
    ($m:ident <= $hqcver:ident) => {{
        use anyhow::anyhow;
        use liboqs_rs_bindings as oqs;
        use oqs::hqc::$hqcver as HQC;
        use oqs::{hqc::Hqc as _, Kem, KemBuf, KemWithRejectionSampling};

        #[pyclass]
        struct $hqcver;

        #[pymethods]
        impl $hqcver {
            #[new]
            fn new() -> Self {
                Self {}
            }

            #[staticmethod]
            fn keypair(py: Python<'_>) -> Result<(&PyBytes, &PyBytes)> {
                let (public, secret) = HQC::keypair().map_err(anyhow::Error::msg)?;

                //makes copies
                Ok((PyBytes::new(
                    py, public.as_slice()
                ), PyBytes::new(
                    py, secret.as_slice()
                )))
            }

            #[staticmethod]
            fn params(what: &str) -> Result<i64> {
                let params = HQC::params();
                match what.to_uppercase().as_str() {
                    "N" => Ok(params.PARAM_N),
                    "N1" => Ok(params.PARAM_N1),
                    "N2" => Ok(params.PARAM_N2),
                    "N1N2" => Ok(params.PARAM_N1N2),
                    "SECURITY" => Ok(params.PARAM_SECURITY),
                    "DELTA" => Ok(params.PARAM_DELTA),
                    _ => Err(anyhow!("No such param!")),
                }
            }

            #[staticmethod]
            fn new_plaintext(py: Python<'_>) -> &PyByteArray {
                PyByteArray::new(
                    py,
                    <HQC as KemWithRejectionSampling>::Plaintext::new().as_slice(),
                )
            }

            /// Extracts x and y from secret key
            #[staticmethod]
            fn secrets_from_key(secretkey: &[u8]) -> Result<(Vec<u64>, Vec<u32>)> {
                let mut sk = <HQC as Kem>::SecretKey::new();
                sk.as_mut_slice().copy_from_slice(secretkey);
                let (x, y) = HQC::secrets_from_key(&sk).map_err(anyhow::Error::msg)?;
                Ok((x.as_slice().to_vec(), y.as_slice().to_vec()))
            }

            /// Returns the number of rejections for the provided plaintext
            /// 
            /// Format: single value with ("number of seed expansions" * 1000 + nbr rejections)
            #[staticmethod]
            fn num_rejections(pt: &[u8]) -> Result<u64> {
                let mut rust_pt = <HQC as KemWithRejectionSampling>::Plaintext::new();
                rust_pt.as_mut_slice().copy_from_slice(pt);
                HQC::num_rejections(&mut rust_pt).map_err(anyhow::Error::msg)
            }

            /// Runs the encapsulation function
            /// 
            /// Returns the ciphertext and shared secret in a tuple
            #[staticmethod]
            fn encaps<'p>(py: Python<'p>, publickey: &[u8]) -> Result<(&'p PyByteArray, &'p PyByteArray)> {
                let mut pk = <HQC as Kem>::PublicKey::new();
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut ss = <HQC as Kem>::SharedSecret::new();
                pk.as_mut_slice().copy_from_slice(publickey);
                HQC::encaps(&mut ct, &mut ss, &mut pk).map_err(anyhow::Error::msg)?;

                Ok((PyByteArray::new(
                        py,
                        ct.as_slice()
                    ),PyByteArray::new(
                        py,
                        ss.as_slice()
                )))
            }


            /// Runs the encapsulation function with provided plaintext and r1 (in a sparse format).
            /// 
            /// Returns the ciphertext and shared secret in a tuple
            #[staticmethod]
            fn encaps_with_plaintext_and_r1<'p>(py: Python<'p>, publickey: &[u8], m: &[u8], mut r1_sparse: Vec<u32>) -> Result<(&'p PyByteArray, &'p PyByteArray)> {
                let mut pk = <HQC as Kem>::PublicKey::new();
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut ss = <HQC as Kem>::SharedSecret::new();
                let mut pt = <HQC as KemWithRejectionSampling>::Plaintext::new();
                pk.as_mut_slice().copy_from_slice(publickey);
                pt.as_mut_slice().copy_from_slice(m);
                HQC::encaps_with_plaintext_and_r1(&mut ct, &mut ss, &mut pk, &mut pt, &mut r1_sparse).map_err(anyhow::Error::msg)?;

                Ok((PyByteArray::new(
                        py,
                        ct.as_slice()
                    ),PyByteArray::new(
                        py,
                        ss.as_slice()
                )))
            }

            /// Extracts the eprime from the ciphertext by decoding it
            #[staticmethod]
            fn eprime<'p>(py: Python<'p>, ciphertext: Vec<u8>, secretkey: &[u8]) -> Result<&'p PyByteArray>{
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut sk = <HQC as Kem>::SecretKey::new();
                ct.as_mut_slice().copy_from_slice(&ciphertext);
                sk.as_mut_slice().copy_from_slice(secretkey);
                let eprime = HQC::eprime(&mut ct, &mut sk).map_err(anyhow::Error::msg)?;
                Ok(PyByteArray::new(
                    py,
                    eprime.as_slice()
                ))
            }
        }

        $m.add_class::<$hqcver>()?;
    }};
}

#[cfg(test)]
mod tests {
    use liboqs_rs_bindings as oqs;
    use oqs::{Kem, hqc::{Hqc128, Hqc}, KemBuf};

    #[test]
    fn hqc_secrets_from_key() {
        let (_pubkey, secretkey) = Hqc128::keypair().unwrap();
        let (x, y) = Hqc128::secrets_from_key(&secretkey).unwrap();
        let mut ones = 0;
        for xi in x.as_slice() {
            ones += xi.count_ones();
        }
        assert_eq!(ones, Hqc128::params().PARAM_OMEGA);
        assert_eq!(y.as_slice().len(), Hqc128::params().PARAM_OMEGA);
    }
}
