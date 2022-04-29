#![allow(clippy::suspicious_arithmetic_impl)]

use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use g2p::GaloisField;
use simulate_rs::Decoder;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

g2p::g2p!(GF16, 4, modulus: 0b10011);

const N: usize = 450;
const R: usize = 150;
const DV: usize = 7;
const DC: usize = 3;

type MyTestDecoder = Decoder<N, R, DV, DC, { GF16::SIZE }, GF16>;

fn h_from_file<P, const N: usize, const R: usize, GF>(path: P) -> Result<[[GF; N]; R]>
where
    P: AsRef<Path>,
    GF: GaloisField + std::convert::From<u8>,
{
    let mut ret = [[GF::ZERO; N]; R];
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    for (row, line) in reader.lines().enumerate() {
        for (column, value) in line?.split_whitespace().enumerate() {
            let int: u8 = value.parse()?;
            ret[row][column] = int.into();
        }
    }

    Ok(ret)
}

fn decoder_benchmark(c: &mut Criterion) {
    let decoder = MyTestDecoder::new(
        h_from_file("simulate-with-python/simulate_rs/benches/parity_check_150_450.txt").unwrap(),
        10,
    );

    // Zero message with zero noise
    let mut channel_output = [[0.0; GF16::SIZE]; N];
    for el in &mut channel_output {
        el[0] = 1.0;
    }

    // Introduce an error
    channel_output[1][0] = 0.1;
    channel_output[1][15] = 0.9;

    // Convert to LLR
    let channel_llr = MyTestDecoder::into_llr(&channel_output);

    c.bench_function("small decoder", |b| {
        b.iter(|| decoder.min_sum(black_box(channel_llr)))
    });
}

criterion_group!(benches, decoder_benchmark);
criterion_main!(benches);
