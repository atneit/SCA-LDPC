#![feature(generic_const_exprs)]
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
use log::debug;

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

/// A variable node
#[derive(Debug, Clone)]
struct VariableNode<const DV: usize, const B: usize> {
    /// options to deal with irregular codes
    check_idx: [Option<Key1D>; DV],
    /// The a-priory channel value
    channel: Option<QaryLlrs<B>>,
}

impl<const DV: usize, const B: usize> Default for VariableNode<DV, B> {
    fn default() -> Self {
        Self {
            check_idx: [None; DV],
            channel: Default::default(),
        }
    }
}

impl<const DV: usize, const B: usize> VariableNode<DV, B> {
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
pub struct QaryLlrs<const B: usize>([FloatType; B]);

// Returns the arguments ordered by value
fn min_max<I: PartialOrd>(in1: I, in2: I) -> (I, I) {
    if in1 < in2 {
        (in1, in2)
    } else {
        (in2, in1)
    }
}

impl<const Q: usize> QaryLlrs<Q> {
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

    // assume hij is 1 or -1, after multiplication array is the same (hij==1) or reversed
    fn mult_in_gf(&self, hij: i8) -> Self {
        let mut ret = Self(self.0);
        if hij < 0 {
            for q in 0..Q {
                ret.0[q] = self.0[Q - q - 1];
            }
        }
        ret
    }

    // q-ary'ily Add self with term multiplied by hij
    fn qary_add_with_mult_in_gf(&self, term: Self, hij: i8) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        if hij > 0 {
            for q in 0..Q {
                ret.0[q] = self.0[q] + term.0[q];
            }
        } else {
            for q in 0..Q {
                ret.0[q] = self.0[q] + term.0[Q - q - 1];
            }
        }
        ret
    }

    // q-ary'ily Subtract self with term multiplied by hij
    fn qary_sub_with_mult_in_gf(&self, subtrahend: Self, hij: i8) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        if hij > 0 {
            for q in 0..Q {
                ret.0[q] = self.0[q] - subtrahend.0[q];
            }
        } else {
            for q in 0..Q {
                ret.0[q] = self.0[q] - subtrahend.0[Q - q - 1];
            }
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

struct SimpleDValueIterator<const SW: usize, BType> {
    b: BType,
    num_ignore: usize,
    d_values: Option<[BType; SW]>,
}

impl<const SW: usize, BType> SimpleDValueIterator<SW, BType>
where
    BType: Copy + Integer + Signed + AddAssign + NumCast,
{
    fn new(b: BType, num_ignore: usize) -> SimpleDValueIterator<SW, BType> {
        assert!(num_ignore <= SW, "num_ignore must be less than or equal to SW");
        Self {
            b,
            num_ignore,
            d_values: None,
        }
    }

    fn increment_d_values(&mut self) -> bool {
        if let Some(ref mut d_values) = self.d_values {
            for i in 0..(SW - self.num_ignore) {
                if d_values[i] < self.b {
                    d_values[i] += BType::from(1_usize).unwrap();
                    return true;
                }
                d_values[i] = -self.b;
            }
            false
        } else {
            let mut initial_values = [-self.b; SW];
            for i in (SW - self.num_ignore)..SW {
                initial_values[i] = BType::zero();
            }
            self.d_values = Some(initial_values);
            true
        }
    }

    pub fn next(&mut self) -> Option<&[BType]> {
        if !self.increment_d_values() {
            return None;
        }

        match &self.d_values {
            Some(ref d_values) => Some(&d_values[0..SW]),
            None => None,
        }
    }
}

type Container2D<T> = rustc_hash::FxHashMap<Key2D, T>;
//type Container2D<T> = HashMap<Key2D, T>;

type ParCheckType = i8;

/// Decoder that implements min_sum algorithm. 
///
/// The generic arguments are:
/// N: Number of variable nodes
/// R: Number of check nodes
/// BVARS: Number of B-variables, must be equal to N-R
/// DC: Maximum check node degree (num variables, per check)
/// DV: Maximum variable node degree (num checks, per variable)
/// B: first N-R variables from the range [-B, ..., 0, ..., B]
/// BSIZE: size of range [-B, ..., 0, ..., B], i.e. 2*B+1
/// BSUM: last R variables from the range [-BSUM, ..., 0, ..., BSUM]
/// BSUMSIZE: size of range [-BSUM, ..., 0, ..., BSUM], i.e. 2*BSUM+1
pub struct DecoderSpecial<
    const N: usize,
    const R: usize,
    const BVARS: usize,
    const SW: usize,
    const DC: usize,
    const DV: usize,
    const B: usize,
    const BSIZE: usize,
    const BSUM: usize,
    const BSUMSIZE: usize,
    BType: Integer + Signed,
> {
    /// Parity check matrix
    parity_check: Container2D<ParCheckType>,
    /// Messages between B-variables and check nodes
    edges: Container2D<Edge<BSIZE>>,
    /// Messages between BSUM-variables and check nodes
    edgessum: Container2D<Edge<BSUMSIZE>>,
    /// List of B-Variable nodes
    vn: [VariableNode<DV, BSIZE>; BVARS],
    /// List of BSUM-Variable nodes
    vnsum: [VariableNode<1, BSUMSIZE>; R],
    /// List of Check nodes, each node contain DC-1 B-variables and 1 BSUM-variable
    cn: [CheckNode<DC>; R],
    /// Number of iterations to perform in the decoder    
    max_iter: u32,
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
        const BVARS: usize,
        const SW: usize,
        const DC: usize,
        const DV: usize,
        const B: usize,
        const BSIZE: usize,
        const BSUM: usize,
        const BSUMSIZE: usize,
        BType,
    > DecoderSpecial<N, R, BVARS, SW, DC, DV, B, BSIZE, BSUM, BSUMSIZE, BType>
where
// TODO: remove std::fmt::Debug
    BType: Integer + Signed + NumCast + AddAssign + Copy + FromPrimitive + TryInto<usize> + Default + std::fmt::Debug,
{
    pub const N: usize = N;
    pub const R: usize = R;
    pub const BVARS: usize = BVARS;
    pub const SW: usize = SW;
    pub const DC: usize = DC;
    pub const DV: usize = DV;
    pub const B: isize = B as isize;
    pub const BSIZE: usize = BSIZE;
    pub const BSUM: isize = BSUM as isize;
    pub const BSUMSIZE: usize = BSUMSIZE;

    pub fn new(parity_check: [[ParCheckType; N]; R], max_iter: u32) -> Self {
        if (BSUM % B) != 0 {
            panic!(
                "BSUM ({}) must be multiple of B ({})", BSUM, B
            );
        }
        if BVARS != N-R {
            panic!("BVARS ({}) must be equal to N-R ({})", BVARS, N-R);
        }
        let vn: RefCell<[VariableNode<DV, BSIZE>; BVARS]> = RefCell::new(make_default_array());
        let vnsum: RefCell<[VariableNode<1, BSUMSIZE>; R]> = RefCell::new(make_default_array());
        let cn: RefCell<[CheckNode<DC>; R]> = RefCell::new(make_default_array());
        let edges = RefCell::new(Container2D::default());
        let edgessum = RefCell::new(Container2D::default());

        let parity_check: Container2D<ParCheckType> = IntoIterator::into_iter(parity_check)
            .enumerate()
            .flat_map(|(row_num, row)| {
                let vn = &vn;
                let vnsum = &vnsum;
                let cn = &cn;
                let edges = &edges;
                let edgessum = &edgessum;
                IntoIterator::into_iter(row)
                    .enumerate()
                    // Filter out zeroes in the parity check
                    .filter(|(col_num, h)| *h != 0)
                    // Handle the non-zeroes
                    .map(move |(col_num, h)| {
                        let ij: Key2D = (row_num, col_num).into();

                        
                        if col_num >= BVARS {
                            // This creates an empty edge (no message in either direction)
                            edgessum.borrow_mut().insert(ij, Default::default());
                            // add the check index to the variable
                            insert_first_none::<_, 1>(
                                &mut vnsum.borrow_mut()[ij.col() - BVARS].check_idx,
                                ij.row,
                            );
                        } else {
                            edges.borrow_mut().insert(ij, Default::default());
                            insert_first_none::<_, DV>(
                                &mut vn.borrow_mut()[ij.col()].check_idx,
                                ij.row,
                            );
                        }

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
        let mut vnsum = vnsum.into_inner();
        let cn = cn.into_inner();
        let edges = edges.into_inner();
        let edgessum = edgessum.into_inner();

        // ::log::info!("Decoder has the following parameters:\n vn={:?}\n\n\n vnsum={:?}\n\n\n cn={:?} \n\n\n edges={:?} \n\n\n edgessum={:?}", vn, vnsum, cn, edges, edgessum);

        Self {
            parity_check,
            vn,
            vnsum,
            cn,
            edges,
            edgessum,
            max_iter,
            brange: ((BType::from(-Self::B).unwrap())..=(BType::from(Self::B).unwrap())),
        }
    }

    /// Formats the sum of two numbers as string.
    ///
    /// # Safety
    ///
    /// This function is safe to use if it does not panic in debug builds!
    pub fn min_sum(&self, channel_llr: [QaryLlrs<BSIZE>; BVARS], channel_llr_sum: [QaryLlrs<BSUMSIZE>; R]) -> Result<[BType; N]> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut vnsum = self.vnsum.clone();
        let mut edges = self.edges.clone();
        let mut edgessum = self.edgessum.clone();
        let mut hard_decision = [BType::zero(); N];

        // 0. Initialize the channel values
        for (var_idx, (v, m)) in (0..).zip(vn.iter_mut().zip(channel_llr)) {
            v.channel = Some(m);
            for key in v.checks(var_idx) {
                // We assume that only plus or minus ones are present in the parity check matrix
                debug_unwrap!(edges.get_mut(&key)).v2c.insert(m.mult_in_gf(self.parity_check[&key]));
            }
        }
        for (var_idx, (v, m)) in ((BVARS as u16)..).zip(vnsum.iter_mut().zip(channel_llr_sum)) {
            v.channel = Some(m);
            for key in v.checks(var_idx) {
                // We assume that only plus or minus ones are present in the parity check matrix
                debug_unwrap!(edgessum.get_mut(&key)).v2c.insert(m.mult_in_gf(self.parity_check[&key]));
            }
        }

        // ::log::info!("Decoder has the following parameters:\n vn={:?}\n\n\n vnsum={:?}\n\n\n cn={:?} \n\n\n edges={:?} \n\n\n edgessum={:?}", vn, vnsum, self.cn, edges, edgessum);
        // ::log::info!("Decoder has the following parameters:\n vn={:?}\n\n\n vnsum={:?}\n\n\n cn={:?} \n\n\n edges={:?} \n\n\n edgessum={:?}", vn[0], vnsum[0], self.cn[0], edges.iter().next(), edgessum.iter().next());

        let mut it = 0;
        'decoding: loop {
            it += 1;
            // 1. Parity check: Compute the syndrome based on the hard_decision
            // noop, we use only num iterations instead
            // 2. check num iterations
            // noop, we do it at the end of the loop instead
            // 3. Check node update (min)
            'check_update: for (check_idx, check) in (0..).zip(&self.cn) {
                // Check nodes in cn are a list of values followed by some amount (potentially 0) of Nones
                // since the code is not generally regular. We assume check matrix is built as H||I,
                // therefore, for all check nodes the last non-None value corresponds to I, i.e. check value 
                let num_nones = check.variable_idx.iter().rev().take_while(|&&x| x.is_none()).count();
                let num_variable_nodes = SW - num_nones;

                let mut check_iter = check.variables(check_idx);
                let alpha_i: Vec<&QaryLlrs<BSIZE>> = check_iter
                    .by_ref()
                    .take(num_variable_nodes)
                    .map(|key| debug_unwrap!(&edges[&key].v2c.as_ref()))
                    .collect();
                let alpha_ij_sum: &QaryLlrs<BSUMSIZE> = check_iter
                    .map(|key| debug_unwrap!(&edgessum[&key].v2c.as_ref()))
                    .next()
                    .unwrap();

                // ::log::info!("alpha_i {:?}, alpha_ij_sum = {:?}", alpha_i, alpha_ij_sum);

                let mut beta_i = [QaryLlrs::<BSIZE>([FloatType::INFINITY; BSIZE]); SW];
                let mut beta_ij_sum = QaryLlrs::<BSUMSIZE>([FloatType::INFINITY; BSUMSIZE]);

                // ::log::info!("beta_i {:?}, beta_ij_sum = {:?}", beta_i, beta_ij_sum);
                
                let mut d_values_iter = SimpleDValueIterator::<SW, BType>::new(BType::from(B).unwrap(), num_nones);
                while let Some(d_values) = d_values_iter.next() {
                    let mut d_value_sum = BType::zero();
                    for d in d_values.iter() {
                        d_value_sum += *d;
                    }
                    d_value_sum = -d_value_sum;

                    let mut sum_of_alpha: FloatType = alpha_i
                        .iter()
                        .zip(d_values)
                        .map(|(alpha_ij, d)| {
                            alpha_ij.0[Self::b2i::<B>(*d)]
                        })
                        .sum();
                    sum_of_alpha += alpha_ij_sum.0[Self::b2i::<BSUM>(d_value_sum)];

                    for ((beta_ij, d), alpha_ij) in beta_i.iter_mut().zip(d_values).zip(&alpha_i) {
                        beta_ij.0[Self::b2i::<B>(*d)] = beta_ij.0[Self::b2i::<B>(*d)]
                            .min(sum_of_alpha - alpha_ij.0[Self::b2i::<B>(*d)]);
                    }
                    beta_ij_sum.0[Self::b2i::<BSUM>(d_value_sum)] = beta_ij_sum.0[Self::b2i::<BSUM>(d_value_sum)]
                        .min(sum_of_alpha - alpha_ij_sum.0[Self::b2i::<BSUM>(d_value_sum)]);
                }

                let mut check_iter = check.variables(check_idx);
                for (key, beta_ij) in check_iter.by_ref().take(num_variable_nodes).zip(beta_i) {
                    debug_unwrap!(edges.get_mut(&key)).c2v.replace(beta_ij);
                }
                debug_unwrap!(edgessum.get_mut(&check_iter.next().unwrap())).c2v.replace(beta_ij_sum);

                // ::log::info!("beta_i {:?}, beta_ij_sum = {:?}", beta_i, beta_ij_sum);
            }

            // Variable node update (sum)
            for (var_idx, var) in (0..).zip(&vn) {
                // Collect connected checks

                // 4.1 primitive messages. Full summation
                let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
                    sum = sum.qary_add_with_mult_in_gf(incoming, self.parity_check[&key]);
                }
                for key in var.checks(var_idx) {
                    // 4.2 primitive outgoing messages, subtract self for individual message
                    let edge = debug_unwrap!(edges.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = sum.qary_sub_with_mult_in_gf(incoming, self.parity_check[&key]).mult_in_gf(self.parity_check[&key]);
                    // 5. Message normalization
                    let arg_min = Self::arg_min::<BSIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    // 6. Tentative decision
                    hard_decision[var_idx as usize] = Self::i2b::<B>(Self::arg_min::<BSIZE>(sum));
                }
            }
            for (var_idx, var) in ((BVARS as u16)..).zip(&vnsum) {
                let mut sum: QaryLlrs<BSUMSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edgessum.get(&key)).c2v);
                    sum = sum.qary_add_with_mult_in_gf(incoming, self.parity_check[&key]);
                }
                for key in var.checks(var_idx) {
                    let edge = debug_unwrap!(edgessum.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = sum.qary_sub_with_mult_in_gf(incoming, self.parity_check[&key]).mult_in_gf(self.parity_check[&key]);
                    let arg_min = Self::arg_min::<BSUMSIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    hard_decision[var_idx as usize] = Self::i2b::<BSUM>(Self::arg_min::<BSUMSIZE>(sum));
                }
            }

            if it >= self.max_iter {
                break 'decoding;
            }
        }

        Ok(hard_decision)
    }

    pub fn into_llr<const T: usize, const U: usize>(channel_output: &[[FloatType; T]; U]) -> [QaryLlrs<T>; U] {
        const EPSILON: FloatType = 0.001;
        let mut llrs = [QaryLlrs::<T>([0.0; T]); U];
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
            assert!(sum < 1.0 + EPSILON);
            assert!(sum > 1.0 - EPSILON);
            // calculate LLR
            for (ent, dst) in var.iter().zip(msg.0.iter_mut()) {
                *dst = (max / ent).ln();
            }
        }

        llrs
    }

    fn arg_min<const T: usize>(m: QaryLlrs<T>) -> usize {
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

    pub fn i2b<const T: usize>(i: usize) -> BType {
        let i: isize = i.try_into().unwrap();
        let b: isize = T.try_into().unwrap();
        let val = i - b;
        if val > b || val < -b {
            panic!("Value over-/underflow!");
        }
        BType::from(val).unwrap()
    }

    pub fn b2i<const T: usize>(val: BType) -> usize
    where
        BType: TryInto<usize>,
    {
        let mut val: isize = val.to_isize().unwrap();

        (val + (T as isize)) as usize
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

    #[test]
    fn into_llr() {
        /* let channel_output = [[
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
        assert_eq!(expected, llr); */
    }

    #[test]
    fn it_works() {
/*         let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new(
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

        assert_eq!(res, expected); */
    }
}
