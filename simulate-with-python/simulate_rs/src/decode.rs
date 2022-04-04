/// Formats the sum of two numbers as string.
pub fn bp_decode_impl(a: usize, b: usize) -> String {
    (a + b).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(bp_decode_impl(2, 3), "5");
    }
}
