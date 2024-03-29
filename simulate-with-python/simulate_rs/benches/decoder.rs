#![allow(clippy::suspicious_arithmetic_impl)]

use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simulate_rs::Decoder;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

const N: usize = 450;
const R: usize = 150;
const DV: usize = 3;
const DC: usize = 7;
const B: usize = 7;
const Q: usize = B * 2 + 1;

type MyTestDecoder = Decoder<N, R, DV, DC, Q, B, i8>;

fn h_from_file<P, const N: usize, const R: usize>(path: P) -> Result<[[i8; N]; R]>
where
    P: AsRef<Path>,
{
    let mut ret = [[0; N]; R];
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    for (row, line) in reader.lines().enumerate() {
        for (column, value) in line?.split_whitespace().enumerate() {
            let int: i8 = value.parse()?;
            ret[row][column] = int;
        }
    }

    Ok(ret)
}

fn medium_decoder_benchmark(c: &mut Criterion) {
    let parity_check = h_from_file("benches/parity_check_150_450.txt")
        .or_else(|_err| {
            h_from_file("simulate-with-python/simulate_rs/benches/parity_check_150_450.txt")
        })
        .unwrap();
    let decoder = MyTestDecoder::new(parity_check, 10);

    // Zero message with zero noise
    let mut channel_output = [[0.0; MyTestDecoder::Q]; MyTestDecoder::N];
    for el in &mut channel_output {
        el[MyTestDecoder::b2i(0)] = 1.0;
    }

    // Introduce an error
    channel_output[1][MyTestDecoder::b2i(0)] = 0.1;
    channel_output[1][MyTestDecoder::b2i(7)] = 0.9;

    // Convert to LLR
    let channel_llr = MyTestDecoder::into_llr(&channel_output);

    c.bench_function("medium decoder", |b| {
        b.iter(|| decoder.min_sum(black_box(channel_llr)))
    });
}

type MyTinyTestDecoder = Decoder<6, 3, 3, 4, 15, 7, i8>;

fn small_decoder_benchmark(c: &mut Criterion) {
    let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new(
        [
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
        ],
        10,
    );

    // Zero message with zero noise
    let mut channel_output = [[0.0; MyTinyTestDecoder::Q]; MyTinyTestDecoder::N];
    for el in &mut channel_output {
        el[MyTinyTestDecoder::b2i(0)] = 1.0;
    }

    // Introduce an error
    channel_output[1][MyTinyTestDecoder::b2i(0)] = 0.1;
    channel_output[1][MyTinyTestDecoder::b2i(7)] = 0.9;

    // Convert to LLR
    let channel_llr = MyTinyTestDecoder::into_llr(&channel_output);

    c.bench_function("small decoder", |b| {
        b.iter(|| decoder_6_3_4_3_gf16.min_sum(black_box(channel_llr)))
    });
}

criterion_group!(benches, small_decoder_benchmark, medium_decoder_benchmark);
criterion_main!(benches);
