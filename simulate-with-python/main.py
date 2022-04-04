#!/bin/env python
"""
Main entry point for the python LDPC code simulator.
"""

__author__ = "Alexander Nilsson; Denis Nabokov; Qian Guo"
__version__ = "0.1.0"
__license__ = "MIT"

# Activate logger
from logzero import logger

logger.debug("Importing dependencies...")

# Import dependencies
from simulate.make_code import (
    make_qc_parity_check_matrix,
    make_regular_ldpc_parity_check_matrix,
    make_regular_ldpc_parity_check_matrix_identity,
)
from simulate.utils import CommandsBase, make_random_state
from simulate.decode import simulate_frame_error_rate, ErrorsProvider
from ldpc import bp_decoder
from ldpc.codes import rep_code
import numpy as np
import argparse
import sys


logger.debug("Importing dependencies... Done")


class Commands(CommandsBase):
    """
    This class defines all possible commands for this simulator to run
    """

    PREFIX = "command_"

    def setup_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--seed",
            action="store",
            type=int,
            required=False,
            help="Use a fixed seed to make simulations reproducible.",
        )
        parser.add_argument(
            "--runs",
            action="store",
            type=int,
            default=100,
            help="The number of runs to run the simulation for. Only relevant for simulations/commands that are non-deterministic.",
        )
        error_group = parser.add_mutually_exclusive_group(required=False)
        error_group.add_argument(
            "--error-rate",
            action="store",
            type=float,
            default=0.05,
            help="The error rate of the simulated binary symmetric channel.",
        )
        error_group.add_argument(
            "--error-file",
            action="store",
            type=str,
            help="Input file specifying distribution of the error for different positions.",
        )

    def command_test_provider(self, args: argparse.Namespace):
        N = 10000
        rng = make_random_state(args.seed)
        errors_provider = ErrorsProvider(0.05, None, rng)
        s = 0
        for i in range(N):
            s += errors_provider.get_error(0)
        print(s/N)

        print('Binary file:')
        errors_provider = ErrorsProvider(0.05, 'binary_distr.txt', rng)
        positions = 4
        r = [0] * positions
        for i in range(positions * N):
            e = errors_provider.get_error(i)
            r[i % positions] += e
        for i in range(positions):
            print(r[i]/N)

        print('q-ary file:')
        errors_provider = ErrorsProvider(0.05, 'qary_distr.txt', rng)
        positions = 2
        r = []
        from collections import defaultdict
        for i in range(positions):
            r.append(defaultdict(int))
        for i in range(positions * N):
            e = errors_provider.get_error(i)
            r[i % positions][e] += 1
        for i in range(positions):
            print('\t'.join(f'{key}: {val/N}' for key, val in r[i].items()))

    def command_regular_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a regular (3,6) ldpc code with a parity check matrix of the form: H_r*k"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        errors_provider = ErrorsProvider(args.error_rate, args.error_file, rng)
        error_rate = args.error_rate
        k = 300  # 17669
        r = 150  #
        rate = k / (k + r)
        row_weight = 6
        column_weight = 3
        # n = k + r
        H = make_regular_ldpc_parity_check_matrix(k, r, column_weight, row_weight, rng)
        logger.info(f"Constructed a rate {rate} code")
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, errors_provider, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_regular_ldpc_code_identity(self, args: argparse.Namespace):
        logger.info(
            "Testing a regular (3,6+1) ldpc code with a parity check matrix of the form: H_r*k|I_r*r"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        error_rate = args.error_rate
        k = 300  # 17669
        r = 150  #
        rate = k / (k + r)
        row_weight = 6
        column_weight = 3
        # n = k + r
        H = make_regular_ldpc_parity_check_matrix_identity(
            k, r, column_weight, row_weight, rng
        )
        logger.info(f"Constructed a rate {rate} code")
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, error_rate, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_qc_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a quasi cyclic ldpc code with a parity check matrix of the form: [H_0|H_1|I]"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        error_rate = args.error_rate
        k = 1000  # 17669
        num_blocks = 2
        r = int(k / num_blocks)
        # n = k + r
        H = make_qc_parity_check_matrix(
            block_len=r, column_weight=3, num_blocks=num_blocks, rng=rng
        )
        logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, error_rate, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_compute_bound(self, args: argparse.Namespace):
        k = 300
        r = 150
        rate = k / (k + r)

        p = args.error_rate
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        # capacity of Binary symmetric channel
        capacity = 1 - entropy
        logger.info(
            f"R {'<' if rate < capacity else '>'} C, where R = {rate}, C = {capacity}"
        )

    def command_official_example(self, args: argparse.Namespace):
        """A simple command to test the functionality of the ldpc package"""
        logger.info("official example")
        rng = make_random_state(args.seed)

        n = 13
        error_rate = args.error_rate
        runs = args.runs
        H = rep_code(n)

        successes = simulate_frame_error_rate(H, error_rate, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")


if __name__ == "__main__":
    """This is executed when run from the command line"""
    cmds = Commands()

    args = cmds.parse_arguments()

    logger.info("Python simulator started")

    cmds.run(args)
