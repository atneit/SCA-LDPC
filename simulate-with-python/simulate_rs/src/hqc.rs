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
            fn name() -> &'static str {
                HQC::NAME
            }

            #[staticmethod]
            fn keypair(py: Python<'_>) -> Result<(&PyBytes, &PyBytes)> {
                let (public, secret) = HQC::keypair().map_err(anyhow::Error::msg)?;

                //makes copies
                Ok((
                    PyBytes::new(py, public.as_slice()),
                    PyBytes::new(py, secret.as_slice()),
                ))
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
                    "OMEGA" => Ok(params.PARAM_OMEGA),
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
            fn encaps<'p>(
                py: Python<'p>,
                publickey: &[u8],
            ) -> Result<(&'p PyByteArray, &'p PyByteArray)> {
                let mut pk = <HQC as Kem>::PublicKey::new();
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut ss = <HQC as Kem>::SharedSecret::new();
                pk.as_mut_slice().copy_from_slice(publickey);
                HQC::encaps(&mut ct, &mut ss, &mut pk).map_err(anyhow::Error::msg)?;

                Ok((
                    PyByteArray::new(py, ct.as_slice()),
                    PyByteArray::new(py, ss.as_slice()),
                ))
            }

            /// Runs the encapsulation function with provided plaintext and r1 (in a sparse format).
            ///
            /// Returns the ciphertext and shared secret in a tuple
            #[staticmethod]
            fn encaps_with_plaintext_and_r1<'p>(
                py: Python<'p>,
                publickey: &[u8],
                m: &[u8],
                mut r1_sparse: Vec<u32>,
            ) -> Result<(&'p PyByteArray, &'p PyByteArray)> {
                let mut pk = <HQC as Kem>::PublicKey::new();
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut ss = <HQC as Kem>::SharedSecret::new();
                let mut pt = <HQC as KemWithRejectionSampling>::Plaintext::new();
                pk.as_mut_slice().copy_from_slice(publickey);
                pt.as_mut_slice().copy_from_slice(m);
                HQC::encaps_with_plaintext_and_r1(
                    &mut ct,
                    &mut ss,
                    &mut pk,
                    &mut pt,
                    &mut r1_sparse,
                )
                .map_err(anyhow::Error::msg)?;

                Ok((
                    PyByteArray::new(py, ct.as_slice()),
                    PyByteArray::new(py, ss.as_slice()),
                ))
            }

            /// Extracts the eprime from the ciphertext by decoding it
            #[staticmethod]
            fn eprime<'p>(
                py: Python<'p>,
                ciphertext: Vec<u8>,
                secretkey: &[u8],
                plaintext: &[u8],
            ) -> Result<&'p PyByteArray> {
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut sk = <HQC as Kem>::SecretKey::new();
                let mut m = <HQC as KemWithRejectionSampling>::Plaintext::new();
                ct.as_mut_slice().copy_from_slice(&ciphertext);
                sk.as_mut_slice().copy_from_slice(secretkey);
                m.as_mut_slice().copy_from_slice(plaintext);
                let eprime = HQC::eprime_m(&mut ct, &mut sk, &mut m).map_err(anyhow::Error::msg)?;
                Ok(PyByteArray::new(py, eprime.as_slice()))
            }

            #[staticmethod]
            fn decode_intermediates<'p>(
                py: Python<'p>,
                ciphertext: Vec<u8>,
                secretkey: &[u8],
            ) -> Result<(
                &'p PyByteArray,
                &'p PyByteArray,
                &'p PyByteArray,
                &'p PyByteArray,
                Vec<u64>,
                Vec<u64>,
            )> {
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut sk = <HQC as Kem>::SecretKey::new();
                ct.as_mut_slice().copy_from_slice(&ciphertext);
                sk.as_mut_slice().copy_from_slice(secretkey);
                let (m, rsencoded, rmdecoded, inp, u, v) =
                    HQC::decode_intermediates(&mut ct, &mut sk).map_err(anyhow::Error::msg)?;
                Ok((
                    PyByteArray::new(py, m.as_slice()),
                    PyByteArray::new(py, rsencoded.as_slice()),
                    PyByteArray::new(py, rmdecoded.as_slice()),
                    PyByteArray::new(py, inp.as_slice()),
                    u.as_slice().to_vec(),
                    v.as_slice().to_vec(),
                ))
            }

            #[staticmethod]
            fn decode_oracle<'p>(
                ciphertext: Vec<u8>,
                secretkey: &[u8],
                num_measurements: u64,
            ) -> Result<Option<u64>> {
                let mut ct = <HQC as Kem>::Ciphertext::new();
                let mut sk = <HQC as Kem>::SecretKey::new();
                let mut ss = <HQC as Kem>::SharedSecret::new();
                ct.as_mut_slice().copy_from_slice(&ciphertext);
                sk.as_mut_slice().copy_from_slice(secretkey);
                
                let mut cpu_core_ident_start = 0u32;
                let mut cpu_core_ident_stop = 0u32;
                let _ = unsafe { __get_cpuid_max(0) }; //Serializing instruction
                let start = unsafe { __rdtscp(&mut cpu_core_ident_start) };
                (0..num_measurements).for_each(|_|{
                    let _ = HQC::decaps(&mut ct, &mut ss, &mut sk); // ignore decapsulation errors
                });
                let stop = unsafe { __rdtscp(&mut cpu_core_ident_stop) };
                let _ = unsafe { __get_cpuid_max(0) }; //Serializing instruction
                if cpu_core_ident_start == cpu_core_ident_stop {
                    Ok(Some((stop - start)/num_measurements)) // same cpuid, probably no context switch
                } else {
                    Ok(None) // different cpuid, probably due to context switch, we discard the measurement
                }
            }
        }

        $m.add_class::<$hqcver>()?;
    }};
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use liboqs_rs_bindings as oqs;
    use oqs::{
        hqc::{Hqc, Hqc128},
        Kem, KemBuf, KemWithRejectionSampling,
    };

    #[test]
    fn hqc_secrets_from_key() {
        let (_pubkey, secretkey) = Hqc128::keypair().unwrap();
        let (x, y) = Hqc128::secrets_from_key(&secretkey).unwrap();
        let mut ones = 0;
        for xi in x.as_slice() {
            ones += xi.count_ones();
        }
        assert_eq!(ones, Hqc128::params::<u32>().PARAM_OMEGA);
        assert_eq!(y.as_slice().len(), Hqc128::params::<usize>().PARAM_OMEGA);
    }

    #[test]
    fn encaps_with_plaintext_and_r1() {
        #![allow(non_snake_case)]
        let N: u32 = Hqc128::params().PARAM_N;

        // keypair
        let (mut pk, mut sk) = Hqc128::keypair().unwrap();

        // extract secrets
        let (_x, mut y) = Hqc128::secrets_from_key(&sk).unwrap();
        y.as_mut_slice().sort_unstable();

        // All zero plaintext
        let mut pt = <Hqc128 as KemWithRejectionSampling>::Plaintext::new();
        let mut ct = <Hqc128 as Kem>::Ciphertext::new();
        let mut ss = <Hqc128 as Kem>::SharedSecret::new();

        {
            let j = 1234;
            let mut r1_sparse = vec![0, j];

            // multiply y with r1

            let mut yyj = vec![];
            yyj.extend(y.as_slice());
            let y_shift_j: Vec<_> = y.as_slice().iter().map(|yi| (*yi + j) % N).collect();
            yyj.extend(y_shift_j);

            yyj.sort();

            let yyj: Vec<_> = yyj
                .into_iter()
                .dedup_with_count()
                .into_iter()
                .filter_map(|(count, i)| if count % 2 == 1 { Some(i) } else { None })
                .collect();

            Hqc128::encaps_with_plaintext_and_r1(
                &mut ct,
                &mut ss,
                &mut pk,
                &mut pt,
                &mut r1_sparse,
            )
            .unwrap();

            let eprime = Hqc128::eprime_m(&mut ct, &mut sk, &mut pt)
                .map_err(anyhow::Error::msg)
                .unwrap();

            let mut eprime_sparse = vec![];
            for (i, byte) in eprime.as_slice().iter().enumerate() {
                for bit in 0..8 {
                    if (byte & (1 << bit)) != 0 {
                        eprime_sparse.push((i*8 + bit) as u32);
                    }
                }
            }

            assert_eq!(yyj, eprime_sparse);
            
        }
    }
}
