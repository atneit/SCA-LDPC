#![allow(unused)]

use std::{
    cell::RefCell,
    collections::HashMap,
    convert::TryInto,
    env::VarError,
    mem::{self, transmute, MaybeUninit},
};

use anyhow::Result;
use g2p::GaloisField;
use ndarray::Array1;
use numpy::ndarray::{ArrayView1, ArrayView2};
use ordered_float::NotNan;

/// A variable node
#[derive(Debug, Clone)]
struct VariableNode<const DC: usize, const Q: usize> {
    /// options to deal with irregular codes
    check_idx: [Option<usize>; DC],
    /// The a-priory channel value
    channel: Option<Message<Q>>,
    /// The current decoded value
    current: Option<Message<Q>>,
}

impl<const DC: usize, const Q: usize> Default for VariableNode<DC, Q> {
    fn default() -> Self {
        Self {
            check_idx: [None; DC],
            channel: Default::default(),
            current: Default::default(),
        }
    }
}

/// A check node
#[derive(Debug)]
struct CheckNode<const DV: usize> {
    /// options to deal with initialization and irregular codes
    variable_idx: [Option<usize>; DV],
}

impl<const DV: usize> Default for CheckNode<DV> {
    fn default() -> Self {
        Self {
            variable_idx: [None; DV],
        }
    }
}

type Llr = f64;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Message<const Q: usize>([Llr; Q]);

#[derive(Debug, Default, Clone)]
struct Edge<const Q: usize> {
    v2c: Option<Message<Q>>,
    c2v: Option<Message<Q>>,
}

type Key1D = u32;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct Key2D {
    row: Key1D,
    col: Key1D,
}

impl Key2D {
    fn row(&self) -> usize {
        self.row as usize
    }
    fn col(&self) -> usize {
        self.col as usize
    }
}

impl From<(usize, usize)> for Key2D {
    fn from(from: (usize, usize)) -> Self {
        Self {
            row: from.0.try_into().expect("Index 1 too large!"),
            col: from.1.try_into().expect("Index 2 too large!"),
        }
    }
}

impl From<(Key1D, Key1D)> for Key2D {
    fn from(from: (Key1D, Key1D)) -> Self {
        Self {
            row: from.0,
            col: from.1,
        }
    }
}

type Container2D<T> = HashMap<Key2D, T>;

struct Decoder<
    const N: usize,
    const R: usize,
    const DV: usize,
    const DC: usize,
    const Q: usize,
    GF: GaloisField,
> {
    parity_check: Container2D<GF>,
    edges: Container2D<Edge<Q>>,
    vn: [VariableNode<DC, Q>; N],
    cn: [CheckNode<DV>; R],
}

fn make_default_array<T: Default, const E: usize>() -> [T; E] {
    // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
    // safe because the type we are claiming to have initialized here is a
    // bunch of `MaybeUninit`s, which do not require initialization.
    let mut data: [MaybeUninit<T>; E] = unsafe { MaybeUninit::uninit().assume_init() };
    // Initialization is safe. If default() allocates any data and if there is
    // a panic during this loop, we have a memory leak, but there is no
    // memory safety issue.
    for elem in &mut data[..] {
        elem.write(Default::default());
    }
    // Everything is initialized. Transmute the array to the
    // initialized type.
    // we want to do the following, but can't until
    // https://github.com/rust-lang/rust/issues/61956 is fixed
    //      unsafe { transmute::<[MaybeUninit<T>; E], [T; E]>(data) }
    //
    // instead we do the following:
    // Using &mut as an assertion of unique "ownership"
    let ptr = &mut data as *mut _ as *mut [T; E];
    let res = unsafe { ptr.read() };
    core::mem::forget(data);
    res
}

fn insert_first_none<T, const E: usize>(array: &mut [Option<T>; E], value: T) {
    for el in array {
        if el.is_none() {
            el.insert(value);
            return;
        }
    }
    panic!("Reached the end of the array, no more space left!")
}

impl<
        const N: usize,
        const R: usize,
        const DV: usize,
        const DC: usize,
        const Q: usize,
        GF: GaloisField,
    > Decoder<N, R, DV, DC, Q, GF>
{
    const N: usize = N;
    const R: usize = R;
    const DV: usize = DV;
    const DC: usize = DC;
    const Q: usize = Q;

    pub fn new(parity_check: [[GF; N]; R]) -> Self {
        let vn: RefCell<[VariableNode<DC, Q>; N]> = RefCell::new(make_default_array());
        let cn: RefCell<[CheckNode<DV>; R]> = RefCell::new(make_default_array());
        let edges = RefCell::new(Container2D::new());

        let parity_check: Container2D<GF> = IntoIterator::into_iter(parity_check)
            .enumerate()
            .flat_map(|(row_num, row)| {
                let vn = &vn;
                let cn = &cn;
                let edges = &edges;
                IntoIterator::into_iter(row)
                    .enumerate()
                    // Filter out zeroes in the parity check
                    .filter(|(col_num, h)| h.ne(&GF::ZERO))
                    // Handle the non-zeroes
                    .map(move |(col_num, h)| {
                        let ij: Key2D = (row_num, col_num).into();

                        // This creates an empty edge (no message in either direction)
                        edges.borrow_mut().insert(ij, Default::default());

                        // add the check index to the variable
                        insert_first_none::<_, DC>(
                            &mut vn.borrow_mut()[ij.col()].check_idx,
                            ij.row(),
                        );

                        // add the variable index to the check
                        insert_first_none::<_, DV>(
                            &mut cn.borrow_mut()[ij.row()].variable_idx,
                            ij.col(),
                        );

                        (ij, h)
                    })
            })
            .collect();

        let mut vn = vn.into_inner();
        let cn = cn.into_inner();
        let edges = edges.into_inner();

        Self {
            parity_check,
            vn,
            cn,
            edges,
        }
    }

    /// Formats the sum of two numbers as string.
    pub fn product_sum(&self, channel_llr: [Message<Q>; N]) -> Result<[GF; N]> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut edges = self.edges.clone();

        // 1. Initialize the channel values
        vn.iter_mut().zip(channel_llr).for_each(move |(v, m)| {
            v.channel = Some(m);
            v.current = None;
        });

        // 2. Check node update
        for (check_idx, check) in self.cn.iter().enumerate() {
            let sign = 1.0;
            let min1 = f64::INFINITY;
            let min2 = f64::INFINITY;
            for var_idx in check.variable_idx.iter().flatten() {
                let var = &vn[*var_idx];
            }
        }

        todo!()
    }

    pub fn into_llr(channel_output: &[[f64; Q]; N]) -> [Message<Q>; N] {
        const EPSILON: f64 = 0.001;
        let mut llrs = [Message::<Q>([0.0; Q]); N];
        for (var, msg) in channel_output.iter().zip(llrs.iter_mut()) {
            let sum: f64 = var.iter().sum();
            let max = var
                .iter()
                .copied()
                // ignore NAN values (errors from the previous line)
                .flat_map(NotNan::new)
                .max()
                .map(NotNan::into_inner)
                .expect("No maximum probability found");
            // Ensure the probabilities sum to 1.0, taking into
            // account the problem of floating point comparisons
            assert!(sum < 1.0 + EPSILON);
            assert!(sum > 1.0 - EPSILON);
            // calculate LLR
            for (ent, dst) in var.iter().zip(msg.0.iter_mut()) {
                *dst = (max / ent).ln();
            }
        }

        llrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    g2p::g2p!(GF16, 4, modulus: 0b10011);

    type MyTinyTestDecoder = Decoder<6, 3, 4, 3, { GF16::SIZE }, GF16>;

    #[test]
    fn into_llr() {
        let channel_output = [[
            0.0, 0.0, 0.0, 0.0, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02, 0.0, 0.0, 0.0, 0.0,
        ]; MyTinyTestDecoder::N];
        let llr = MyTinyTestDecoder::into_llr(&channel_output);
        let expected = [Message([
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.9459101490553135,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ]); MyTinyTestDecoder::N];
        assert_eq!(expected, llr);
    }

    #[test]
    fn it_works() {
        let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new([
            [1.into(), 1.into(), 1.into(), 1.into(), 0.into(), 0.into()],
            [0.into(), 0.into(), 1.into(), 1.into(), 0.into(), 1.into()],
            [1.into(), 0.into(), 0.into(), 1.into(), 1.into(), 0.into()],
        ]);

        // Zero message with zero noise
        let mut channel_output = [[0.0; MyTinyTestDecoder::Q]; MyTinyTestDecoder::N];
        for el in &mut channel_output {
            el[0] = 1.0;
        }

        // Introduce an error
        channel_output[1][0] = 0.1;
        channel_output[1][15] = 0.9;

        // Convert to LLR
        let channel_llr = MyTinyTestDecoder::into_llr(&channel_output);

        let res = decoder_6_3_4_3_gf16
            .product_sum(channel_llr)
            .expect("Failed");

        let expected: [GF16; 6] = [0.into(); 6];

        assert_eq!(res, expected);
    }
}
