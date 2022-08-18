#![allow(unused)]

use anyhow::Result;
use fastcmp::Compare;
use g2p::GaloisField;
use ndarray::Array1;
use num::{Float, FromPrimitive, Integer, NumCast, Signed, ToPrimitive};
use numpy::ndarray::{ArrayView1, ArrayView2};
use ordered_float::NotNan;
use std::{
    cell::RefCell,
    cmp::min,
    collections::HashMap,
    convert::{TryFrom, TryInto},
    env::VarError,
    marker::PhantomData,
    mem::{self, transmute, MaybeUninit},
    ops::{Add, AddAssign, Range, RangeInclusive},
};

macro_rules! debug_unwrap {
    ($what:expr) => {{
        #[cfg(debug_assertions)]
        {
            $what.unwrap()
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            $what.unwrap_unchecked()
        }
    }};
}

// TODO: change Q
/// A variable node
#[derive(Debug, Clone)]
struct VariableNode<const DV: usize, const Q: usize> {
    /// options to deal with irregular codes
    check_idx: [Option<Key1D>; DV],
    /// The a-priory channel value
    channel: Option<QaryLlrs<Q>>,
}

impl<const DV: usize, const Q: usize> Default for VariableNode<DV, Q> {
    fn default() -> Self {
        Self {
            check_idx: [None; DV],
            channel: Default::default(),
        }
    }
}

impl<const DV: usize, const Q: usize> VariableNode<DV, Q> {
    fn checks(&self, var_idx: Key1D) -> impl Iterator<Item = Key2D> + '_ {
        self.check_idx
            .iter()
            .flatten()
            .map(move |check_idx| (*check_idx, var_idx).into())
    }
}

/// A check node
#[derive(Debug)]
struct CheckNode<const DC: usize> {
    /// options to deal with initialization and irregular codes
    variable_idx: [Option<Key1D>; DC],
}

impl<const DC: usize> Default for CheckNode<DC> {
    fn default() -> Self {
        Self {
            variable_idx: [None; DC],
        }
    }
}

impl<const DC: usize> CheckNode<DC> {
    fn variables(&self, check_idx: Key1D) -> impl Iterator<Item = Key2D> + '_ {
        self.variable_idx
            .iter()
            .flatten()
            .map(move |var_idx| (check_idx, *var_idx).into())
    }
}

pub type FloatType = f32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QaryLlrs<const Q: usize>([FloatType; Q]);

// Returns the arguments ordered by value
fn min_max<I: PartialOrd>(in1: I, in2: I) -> (I, I) {
    if in1 < in2 {
        (in1, in2)
    } else {
        (in2, in1)
    }
}

impl<const Q: usize> QaryLlrs<Q> {
    /// Runs a three-way q-ary min-function, which returns the 2 minimum values of itself and two arguments
    fn qary_3min2(&self, incoming1: Self, incoming2: Self) -> (Self, Self) {
        let mut min1 = Self([FloatType::INFINITY; Q]);
        let mut min2 = Self([FloatType::INFINITY; Q]);
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
        let mut ret = Self([FloatType::INFINITY; Q]);
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
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] + term.0[q];
        }
        ret
    }

    // q-ary'ily Subtract self with subtrahend
    fn qary_sub(&self, subtrahend: Self) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - subtrahend.0[q];
        }
        ret
    }

    fn qary_sub_arg(&self, arg_min: usize) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - self.0[arg_min];
        }
        ret
    }
}

#[derive(Debug, Default, Clone)]
struct Edge<const Q: usize> {
    v2c: Option<QaryLlrs<Q>>,
    c2v: Option<QaryLlrs<Q>>,
}

type Key1D = u16;

#[allow(clippy::derive_hash_xor_eq)]
#[derive(Debug, Eq, Hash, Clone, Copy)]
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

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

impl PartialEq for Key2D {
    fn eq(&self, other: &Self) -> bool {
        let self_bytes = unsafe { any_as_u8_slice(self) };
        let other_bytes = unsafe { any_as_u8_slice(other) };
        self_bytes.feq(other_bytes)
    }
}

impl From<(usize, usize)> for Key2D {
    fn from(from: (usize, usize)) -> Self {
        unsafe {
            Self {
                row: debug_unwrap!(from.0.try_into()),
                col: debug_unwrap!(from.1.try_into()),
            }
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

#[derive(Debug)]
struct Configuration<const DC: usize, BType> {
    sum: FloatType,
    d_values: [BType; DC],
    alpha_i: [FloatType; DC],
}

impl<const DC: usize, BType: Default + Copy> Default for Configuration<DC, BType> {
    fn default() -> Self {
        Self {
            sum: Default::default(),
            d_values: [BType::default(); DC],
            alpha_i: [0.0; DC],
        }
    }
}

/// (fake) Iterator that finds all possible configurations.
/// Performance is improved greatly by only trying d_values
/// which are possible according to alpha values
struct FiniteDValueIterator<const DC: usize, const Q: usize, BType> {
    finite_d_values: [[BType; Q]; DC],
    num: [usize; DC],
    len: usize,
    indices: Option<[usize; DC]>,
    d_values: Option<[BType; DC]>,
}

impl<const DC: usize, const Q: usize, BType> FiniteDValueIterator<DC, Q, BType>
where
    BType: Copy + Integer + Signed + AddAssign,
{
    fn new<F: Fn(usize) -> BType>(
        alpha_i: &[&QaryLlrs<Q>],
        i2b: F,
    ) -> FiniteDValueIterator<DC, Q, BType> {
        let mut cfg_iter = FiniteDValueIterator {
            finite_d_values: [[BType::zero(); Q]; DC],
            num: [0; DC],
            len: 0,
            indices: None,
            d_values: None,
        };

        for (j, alpha_ij) in alpha_i.iter().enumerate() {
            for (d, alpha_ijd) in alpha_ij.0.iter().enumerate() {
                if alpha_ijd.is_finite() {
                    cfg_iter.finite_d_values[j][cfg_iter.num[j]] = i2b(d);
                    cfg_iter.num[j] += 1;
                }
            }
            cfg_iter.len += 1;
        }

        cfg_iter
    }

    /// Updates d_values according to indices.
    /// Returns whether or not the configuration is valid (summation of b_values = 0)
    fn update_d_values(&mut self) -> bool {
        if let Some(indices) = self.indices {
            if self.d_values.is_none() {
                self.d_values.insert([BType::zero(); DC]);
            }
            let d_values = debug_unwrap!(self.d_values.as_mut());
            let mut dsum = BType::zero();
            for ((idx, d), available) in indices[0..self.len - 1]
                .iter()
                .zip(d_values)
                .zip(self.finite_d_values)
            {
                *d = available[*idx];
                dsum += *d;
            }
            // if -dsum is a possible value, then the configuration is valid
            let available = self.finite_d_values[self.len - 1];
            if available.contains(&(-dsum)) {
                let d_values = debug_unwrap!(self.d_values.as_mut());
                d_values[self.len - 1] = -dsum;
                true
            } else {
                false
            }
        } else {
            // Empty d_values
            self.d_values.take();
            false
        }
    }

    /// Increments the indices such that all permutations are reached
    /// Returns if the increment was successful or not
    fn increment_indices(&mut self) -> bool {
        if let Some(ref mut indices) = self.indices {
            // we have a state, let's update it
            for (idx, num) in indices[0..self.len - 1].iter_mut().zip(self.num) {
                if *idx + 1 < num {
                    // We increase the index, because we can
                    *idx += 1;
                    return true; // nothing more to be done
                }
                // we move on to the next index, but reset the current one first
                *idx = 0;
            }

            // we failed to increment
            false
        } else {
            // we don't have a state, let's start with the [0, 0,..., 0] state
            // but first check that it's a valid state
            if !self.num[0..self.len].contains(&0) {
                self.indices = Some([0; DC])
            }
            true
        }
    }

    /// Returns the next valid configuration or None.
    ///
    /// Emulates an iterator, but does not implement a real iterator
    /// because Iterator does not allow for references into itself due
    /// to conflicting lifetime requirements
    pub fn next(&mut self) -> Option<&[BType]> {
        loop {
            // Update indices
            if !self.increment_indices() {
                return None; // No more configurations left
            }
            // if valid, use it, otherwise find next
            if self.update_d_values() {
                break;
            }
        }

        // return reference to the current d_values
        match &self.d_values {
            Some(ref d_values) => Some(&d_values[0..self.len]),
            None => None,
        }
    }
}

type Container2D<T> = rustc_hash::FxHashMap<Key2D, T>;
//type Container2D<T> = HashMap<Key2D, T>;

type ParCheckType = bool;

/// Decoder that implements min_sum algorithm.
///
/// The generic arguments are:
/// N: Number of variable nodes
/// R: Number of check nodes
/// DC: Maximum check node degree (num variables, per check)
/// DV: Maximum variable node degree (num checks, per variable)
/// Q: Q-ary, that is number Q sub messages in the tanner graph, per edge
/// B: value mapping [-B, ..., 0, ..., B] where Q == 2B+1
pub struct Decoder<
    const N: usize,
    const R: usize,
    const DC: usize,
    const DV: usize,
    const Q: usize,
    const B: usize,
    BType: Integer + Signed,
> {
    /// Parity check matrix
    parity_check: Container2D<ParCheckType>,
    /// Messages between variables and check nodes
    edges: Container2D<Edge<Q>>,
    /// List of Variable nodes
    vn: [VariableNode<DV, Q>; N],
    /// List of Check nodes
    cn: [CheckNode<DC>; R],
    /// Number of iterations to perform in the decoder    
    max_iter: u32,
    /// The range of valid values [-B, ..., 0, ..., B]
    brange: RangeInclusive<BType>,
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
        const DC: usize,
        const DV: usize,
        const Q: usize,
        const B: usize,
        BType,
    > Decoder<N, R, DC, DV, Q, B, BType>
where
    BType: Integer + Signed + NumCast + AddAssign + Copy + FromPrimitive + TryInto<usize> + Default,
{
    pub const N: usize = N;
    pub const R: usize = R;
    pub const DC: usize = DC;
    pub const DV: usize = DV;
    pub const Q: usize = Q;
    pub const B: isize = B as isize;

    pub fn new(parity_check: [[ParCheckType; N]; R], max_iter: u32) -> Self {
        if B * 2 + 1 > Q {
            let b = B as i32;
            let bb: Vec<i32> = (-b..=b).collect();
            panic!(
                "Cannot instantiate decoder with Q={} sub-messages and provided mapping {:?}",
                Q, bb
            );
        }
        let vn: RefCell<[VariableNode<DV, Q>; N]> = RefCell::new(make_default_array());
        let cn: RefCell<[CheckNode<DC>; R]> = RefCell::new(make_default_array());
        let edges = RefCell::new(Container2D::default());

        let parity_check: Container2D<ParCheckType> = IntoIterator::into_iter(parity_check)
            .enumerate()
            .flat_map(|(row_num, row)| {
                let vn = &vn;
                let cn = &cn;
                let edges = &edges;
                IntoIterator::into_iter(row)
                    .enumerate()
                    // Filter out zeroes in the parity check
                    .filter(|(col_num, h)| *h)
                    // Handle the non-zeroes
                    .map(move |(col_num, h)| {
                        let ij: Key2D = (row_num, col_num).into();

                        // This creates an empty edge (no message in either direction)
                        edges.borrow_mut().insert(ij, Default::default());

                        // add the check index to the variable
                        insert_first_none::<_, DV>(
                            &mut vn.borrow_mut()[ij.col()].check_idx,
                            ij.row,
                        );

                        // add the variable index to the check
                        insert_first_none::<_, DC>(
                            &mut cn.borrow_mut()[ij.row()].variable_idx,
                            ij.col,
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
            brange: ((BType::from(-Self::B).unwrap())..=(BType::from(Self::B).unwrap())),
        }
    }

    /// Formats the sum of two numbers as string.
    ///
    /// # Safety
    ///
    /// This function is safe to use if it does not panic in debug builds!
    pub fn min_sum(&self, channel_llr: [QaryLlrs<Q>; N]) -> Result<[BType; N]> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut edges = self.edges.clone();
        let mut hard_decision = [BType::zero(); N];

        // 0. Initialize the channel values
        for (var_idx, (v, m)) in (0..).zip(vn.iter_mut().zip(channel_llr)) {
            v.channel = Some(m);
            for key in v.checks(var_idx) {
                // We assume that only ones are present in the parity check matrix
                debug_unwrap!(edges.get_mut(&key)).v2c.insert(m);
            }
        }

        let mut it = 0;
        // Fill array with -B values
        let mut configurations = Vec::<Configuration<DC, BType>>::new();
        'decoding: loop {
            it += 1;
            // 1. Parity check: Compute the syndrome based on the hard_decision
            // noop, we use only num iterations instead
            // 2. check num iterations
            // noop, we do it at the end of the loop instead
            // 3. Check node update (min)
            'check_update: for (check_idx, check) in (0..).zip(&self.cn) {
                {
                    let alpha_i: Vec<&QaryLlrs<Q>> = check
                        .variables(check_idx)
                        .map(|key| debug_unwrap!(&edges[&key].v2c.as_ref()))
                        .collect();

                    //println!("it: {}, check_idx: {}", it, check_idx);
                    let mut finite_d_values =
                        FiniteDValueIterator::<DC, Q, BType>::new(&alpha_i, Self::i2b);
                    while let Some(d_values) = finite_d_values.next() {
                        // This is a valid configuration where the final d_value is set
                        // to counterweight dsum (to make parity check succeed).
                        // Generate configuration
                        let mut cfg = Configuration::default();
                        cfg.sum = alpha_i
                            .iter()
                            .zip(d_values)
                            .zip(&mut cfg.alpha_i)
                            .zip(&mut cfg.d_values)
                            .map(|(((src_alpha_ij, src_d), dst_alpha_ijd), dst_d)| {
                                *dst_alpha_ijd = src_alpha_ij.0[Self::b2i(*src_d)];
                                *dst_d = *src_d;
                                *dst_alpha_ijd
                            })
                            .sum();
                        // We check if the configuration is possible based on alpha values
                        if cfg.sum.is_finite() {
                            configurations.push(cfg)
                        }
                    }
                }

                assert!(!configurations.is_empty());

                // Find and send minimum Llrs to variables
                for (j, key) in check.variables(check_idx).enumerate() {
                    let mut beta_ij = QaryLlrs::<Q>([FloatType::INFINITY; Q]);
                    for cfg in &configurations {
                        let beta_ijd = &mut beta_ij.0[Self::b2i(cfg.d_values[j])];
                        let cfg_sum = cfg.sum - cfg.alpha_i[j];
                        *beta_ijd = cfg_sum.min(*beta_ijd);
                    }
                    debug_unwrap!(edges.get_mut(&key)).c2v.replace(beta_ij);
                }
                configurations.clear();
            }

            // Variable node update (sum)
            for (var_idx, var) in (0..).zip(&vn) {
                // Collect connected checks

                // 4.1 primitive messages. Full summation
                let mut sum: QaryLlrs<Q> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
                    sum = sum.qary_add(incoming)
                }
                for key in var.checks(var_idx) {
                    // 4.2 primitive outgoing messages, subtract self for individual message
                    let edge = debug_unwrap!(edges.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = sum.qary_sub(incoming);
                    // 5. Message normalization
                    let arg_min = Self::arg_min(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    // 6. Tentative decision
                    hard_decision[var_idx as usize] = Self::i2b(Self::arg_min(sum));
                }
            }

            if it >= self.max_iter {
                break 'decoding;
            }
        }

        Ok(hard_decision)
    }

    pub fn into_llr(channel_output: &[[FloatType; Q]; N]) -> [QaryLlrs<Q>; N] {
        const EPSILON: FloatType = 0.001;
        let mut llrs = [QaryLlrs::<Q>([0.0; Q]); N];
        for (var, msg) in channel_output.iter().zip(llrs.iter_mut()) {
            let sum: FloatType = var.iter().sum();
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
            assert!(sum < 1.0 + EPSILON, "{} < 1.0 - EPSILON", sum);
            assert!(sum > 1.0 - EPSILON, "{} > 1.0 - EPSILON", sum);
            // calculate LLR
            for (ent, dst) in var.iter().zip(msg.0.iter_mut()) {
                *dst = (max / ent).ln();
            }
        }

        llrs
    }

    fn arg_min(m: QaryLlrs<Q>) -> usize {
        let mut min_val = FloatType::INFINITY;
        let mut min_arg = 0;
        for (arg, val) in m.0.iter().copied().enumerate() {
            if val < min_val {
                min_val = val;
                min_arg = arg;
            }
        }
        min_arg
    }

    pub fn i2b(i: usize) -> BType {
        let i: isize = i.try_into().unwrap();
        let b: isize = B.try_into().unwrap();
        let val = i - b;
        if val > b || val < -b {
            panic!("Value over-/underflow!");
        }
        BType::from(val).unwrap()
    }

    pub fn b2i(val: BType) -> usize
    where
        BType: TryInto<usize>,
    {
        let mut val: isize = val.to_isize().unwrap();

        (val + (B as isize)) as usize
    }
}

fn into_or_panic<T, U>(from: T) -> U
where
    T: TryInto<U>,
{
    match from.try_into() {
        Ok(val) => val,
        Err(_) => panic!("Failed conversion!"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    g2p::g2p!(GF16, 4, modulus: 0b10011);

    type MyTinyTestDecoder = Decoder<6, 3, 4, 3, 15, 7, i8>;

    #[test]
    fn into_llr() {
        let channel_output = [[
            0.0, 0.0, 0.0, 0.0, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02, 0.0, 0.0, 0.0,
        ]; MyTinyTestDecoder::N];
        let llr = MyTinyTestDecoder::into_llr(&channel_output);
        let expected = [QaryLlrs([
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            //1.9459101490553135,
            1.945_910_1,
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
        ]); MyTinyTestDecoder::N];
        assert_eq!(expected, llr);
    }

    #[test]
    fn it_works() {
        let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new(
            [
                [true, true, true, true, false, false],
                [false, false, true, true, false, true],
                [true, false, false, true, true, false],
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

        let res = decoder_6_3_4_3_gf16.min_sum(channel_llr).expect("Failed");

        let expected: [i8; 6] = [0; 6];

        assert_eq!(res, expected);
    }
}
