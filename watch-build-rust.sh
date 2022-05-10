#!/bin/sh


cargo watch -s "cargo test --target-dir ../../target/test" -s "maturin develop" -C "simulate-with-python/simulate_rs/"
