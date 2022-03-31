use anyhow::Result;
use ldpc::{codes::LinearCode, decoders::LinearDecoder, noise::Probability};
use rand::thread_rng;
use sparse_bin_mat::SparseBinVec;

fn main() -> Result<()> {
    let num_iterations = 100;
    let probability = Probability::new(0.001);
    let code = code()?;
    let channel = ldpc::noise::BinarySymmetricChannel::with_probability(probability);
    let original_message = SparseBinVec::new(1000, vec![0, 2, 4, 8, 16]);

    let error = code.random_error(&channel, &mut thread_rng());
    let received_message = &original_message + &error;

    let decoder =
        ldpc::decoders::BpDecoder::new(code.parity_check_matrix(), probability, num_iterations);

    let re = decoder.decode(received_message.as_view());

    println!(
        "original_message: {:?}, error: {:?}, received_message: {:?}, re: {:?}",
        original_message, error, received_message, re
    );

    Ok(())
    /*
    let decoder = decoder(&code, probability, max_iterations);
    let noise = DepolarizingNoise::with_probability(probability);
    let mut failures = 0;
    for i in 0..100 {
        println!("{i}");
        let error = code.random_error(&noise, &mut thread_rng());
        let syndrome = code.syndrome_of(&error);
        let correction: PauliOperator = decoder.correction_for(syndrome.as_view()).into();
        if !code.has_stabilizer(&(&error * &correction)) {
            failures += 1;
        }
    }
    println!("{}", failures); */
}

fn code() -> Result<LinearCode> {
    Ok(LinearCode::random_regular_code()
        .num_bits(1000)
        .num_checks(1000)
        .bit_degree(3)
        .check_degree(3)
        .sample_with(&mut thread_rng())
        .unwrap())
}
