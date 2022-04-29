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
}

impl<const DC: usize, const Q: usize> Default for VariableNode<DC, Q> {
    fn default() -> Self {
        Self {
            check_idx: [None; DC],
            channel: Default::default(),
        }
    }
}

impl<const DC: usize, const Q: usize> VariableNode<DC, Q> {
    fn checks(&self, var_idx: usize) -> impl Iterator<Item = Key2D> + '_ {
        self.check_idx
            .iter()
            .flatten()
            .map(move |check_idx| (*check_idx, var_idx).into())
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

impl<const DV: usize> CheckNode<DV> {
    fn variables(&self, check_idx: usize) -> impl Iterator<Item = Key2D> + '_ {
        self.variable_idx
            .iter()
            .flatten()
            .map(move |var_idx| (check_idx, *var_idx).into())
    }
}

type Llr = f64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Message<const Q: usize>([Llr; Q]);

// Returns the arguments ordered by value
fn min_max<I: PartialOrd>(in1: I, in2: I) -> (I, I) {
    if in1 < in2 {
        (in1, in2)
    } else {
        (in2, in1)
    }
}

impl<const Q: usize> Message<Q> {
    /// Runs a three-way q-ary min-function, which returns the 2 minimum values of itself and two arguments
    fn qary_3min2(&self, incoming1: Self, incoming2: Self) -> (Self, Self) {
        let mut min1 = Self([Llr::INFINITY; Q]);
        let mut min2 = Self([Llr::INFINITY; Q]);
        for q in 0..Q {
            let t1 = self.0[q];
            let t2 = incoming1.0[q];
            let t3 = incoming2.0[q];

            // sorting network
            let (t1, t2) = min_max(t1, t2); // min of 1-2
            let (t1, t3) = min_max(t1, t3); // min (1-2)-3
            let (t2, t3) = min_max(t2, t3); // secondary min

            // save min1 and min2, discard min3
            min1.0[q] = t1;
            min2.0[q] = t2;
        }

        (min1, min2)
    }

    /// q-ary'ily returns the given primary value, unless self equals primary, in which
    /// case we return the secondary value.
    fn qary_get_unequal(&self, primary: Self, secondary: Self) -> Self {
        let mut ret = Self([Llr::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = if self.0[q] != primary.0[q] {
                primary.0[q]
            } else {
                secondary.0[q]
            };
        }
        ret
    }

    // q-ary'ily Add self with term
    fn qary_add(&self, term: Self) -> Self {
        let mut ret = Self([Llr::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] + term.0[q];
        }
        ret
    }

    // q-ary'ily Subtract self with subtrahend
    fn qary_sub(&self, subtrahend: Self) -> Self {
        let mut ret = Self([Llr::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - subtrahend.0[q];
        }
        ret
    }

    fn qary_sub_arg(&self, arg_min: usize) -> Self {
        let mut ret = Self([Llr::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - self.0[arg_min];
        }
        ret
    }
}

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

type Container2D<T> = rustc_hash::FxHashMap<Key2D, T>;
//type Container2D<T> = HashMap<Key2D, T>;

pub struct Decoder<
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
    max_iter: u32,
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

    pub fn new(parity_check: [[GF; N]; R], max_iter: u32) -> Self {
        let vn: RefCell<[VariableNode<DC, Q>; N]> = RefCell::new(make_default_array());
        let cn: RefCell<[CheckNode<DV>; R]> = RefCell::new(make_default_array());
        let edges = RefCell::new(Container2D::default());

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
            max_iter,
        }
    }

    /// Formats the sum of two numbers as string.
    pub fn min_sum(&self, channel_llr: [Message<Q>; N]) -> Result<[GF; N]> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut edges = self.edges.clone();
        let mut hard_decision = [GF::ZERO; N];

        // 0. Initialize the channel values
        for (var_idx, (v, m)) in vn.iter_mut().zip(channel_llr).enumerate() {
            v.channel = Some(m);
            for key in v.checks(var_idx) {
                // We assume that only ones are present in the parity check matrix
                assert!(self.parity_check.get(&key).unwrap() == &GF::ONE);
                edges
                    .get_mut(&key)
                    .expect("(Initialization) Edge missing in tanner graph, this is a bug!")
                    .v2c
                    .insert(m);
            }
        }

        let mut it = 0;
        loop {
            it += 1;
            // 1. Parity check: Compute the syndrome based on the hard_decision
            // noop, we use only num iterations instead
            // 2. check num iterations
            // noop, we do it at the end of the loop instead
            // 3. Check node update (min)
            for (check_idx, check) in self.cn.iter().enumerate() {
                let mut min1 = Message([f64::INFINITY; Q]);
                let mut min2 = Message([f64::INFINITY; Q]);

                // 3.1 Find min1 and min2 values
                for key in check.variables(check_idx) {
                    //let key = (check_idx, var_idx).into();
                    (min1, min2) = edges
                        .get(&key)
                        .expect("(Check update) Edge missing in tanner graph, this is a bug!")
                        .v2c
                        .expect("No incoming message for check node!")
                        .qary_3min2(min1, min2);
                }

                // 3.1 Send check messages back to variable node
                for key in check.variables(check_idx) {
                    //let key = (check_idx, var_idx).into();
                    let edge = edges
                        .get_mut(&key)
                        .expect("(Check update 2) Edge missing in tanner graph, this is a bug!");
                    edge.c2v.replace(
                        edge.v2c
                            .expect("No incoming message for check node!")
                            .qary_get_unequal(min1, min2),
                    );
                }
            }

            // Variable node update (sum)
            for (var_idx, var) in vn.iter().enumerate() {
                // Collect connected checks

                // 4.1 primitive messages. Full summation
                let mut sum: Message<Q> = var.channel.expect("Missing channel value!");
                for key in var.checks(var_idx) {
                    let incoming = edges
                        .get(&key)
                        .expect("(Variable update 1) Edge missing in tanner graph, this is a bug!")
                        .c2v
                        .expect("No incoming message for variable node!");
                    sum = sum.qary_add(incoming)
                }
                for key in var.checks(var_idx) {
                    // 4.2 primitive outgoing messages, subtract self for individual message
                    let edge = edges
                        .get_mut(&key)
                        .expect("(Variable update 1) Edge missing in tanner graph, this is a bug!");
                    let incoming = edge.c2v.expect("No incoming message for variable node!");
                    let prim_out = sum.qary_sub(incoming);
                    // 5. Message normalization
                    let arg_min = arg_min(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    // 6. Tentative decision
                    hard_decision[var_idx] = arg_min_gf(sum);
                }
            }

            if it >= self.max_iter {
                break;
            }
        }

        Ok(hard_decision)
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

fn arg_min<const Q: usize>(m: Message<Q>) -> usize {
    let mut min_val = Llr::INFINITY;
    let mut min_arg = 0;
    for (arg, val) in m.0.iter().copied().enumerate() {
        if val < min_val {
            min_val = val;
            min_arg = arg;
        }
    }
    min_arg
}

fn arg_min_gf<const Q: usize, GF: GaloisField>(m: Message<Q>) -> GF {
    let mut min_val = Llr::INFINITY;
    let mut min_arg = GF::ZERO;
    let mut arg = GF::ZERO;
    for val in m.0.iter().copied() {
        if val < min_val {
            min_val = val;
            min_arg = arg;
        }
        arg += GF::ONE;
    }
    min_arg
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
        let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new(
            [
                [1.into(), 1.into(), 1.into(), 1.into(), 0.into(), 0.into()],
                [0.into(), 0.into(), 1.into(), 1.into(), 0.into(), 1.into()],
                [1.into(), 0.into(), 0.into(), 1.into(), 1.into(), 0.into()],
            ],
            10,
        );

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

        let res = decoder_6_3_4_3_gf16.min_sum(channel_llr).expect("Failed");

        let expected: [GF16; 6] = [0.into(); 6];

        assert_eq!(res, expected);
    }
}
