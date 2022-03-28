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
)
from simulate.utils import CommandsBase, pretty_string_matrix
from simulate.decode import simulate_frame_error_rate
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
        parser.add_argument(
            "--error-rate",
            action="store",
            type=float,
            default=0.05,
            help="The error rate of the simulated binary symmetric channel.",
        )

    def command_regular_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a quasi cyclic ldpc code with a parity check matrix of the form: [H_0|H_1|I]"
        )
        runs = args.runs
        error_rate = args.error_rate
        k = 100  # 17669
        rate = 0.5
        r = int(k * rate)
        row_weight = 6
        column_weight = int(row_weight * rate)
        # n = k + r
        H = make_regular_ldpc_parity_check_matrix(k, r, column_weight, row_weight)
        with np.printoptions(threshold=sys.maxsize):
            logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, error_rate, runs)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_qc_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a quasi cyclic ldpc code with a parity check matrix of the form: [H_0|H_1|I]"
        )
        runs = args.runs
        error_rate = args.error_rate
        k = 1000  # 17669
        num_blocks = 2
        r = int(k / num_blocks)
        # n = k + r
        H = make_qc_parity_check_matrix(
            block_len=r, column_weight=3, num_blocks=num_blocks
        )
        logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, error_rate, runs)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_official_example(self, args: argparse.Namespace):
        """A simple command to test the functionality of the ldpc package"""
        logger.info("official example")

        n = 13
        error_rate = args.error_rate
        runs = args.runs
        H = rep_code(n)

        successes = simulate_frame_error_rate(H, error_rate, runs)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")


if __name__ == "__main__":
    """This is executed when run from the command line"""
    cmds = Commands()

    args = cmds.parse_arguments()

    logger.info("Python simulator started")

    cmds.run(args)
