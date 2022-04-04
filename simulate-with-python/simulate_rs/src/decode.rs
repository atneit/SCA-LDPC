use anyhow::Result;
use ndarray::Array1;
use numpy::ndarray::{ArrayView1, ArrayView2};

/// Formats the sum of two numbers as string.
pub fn product_sum<A: Default + Clone>(
    _parity_check: ArrayView2<A>,
    _error_rate: ArrayView1<f64>,
    _max_iter: usize,
    _syndrome: ArrayView1<A>,
) -> Result<Array1<A>> {
    Ok(Array1::from_elem(
        (_error_rate.shape()[0],),
        Default::default(),
    ))
}

#[cfg(test)]
mod tests {
    //use super::*;
    #[test]
    fn it_works() {
        //assert_eq!(product_sum(2, 3), "5");
    }
}
