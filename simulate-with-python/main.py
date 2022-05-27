#!/bin/env python
"""
Main entry point for the python LDPC code simulator.
"""

__author__ = "Alexander Nilsson; Denis Nabokov; Qian Guo"
__version__ = "0.1.0"
__license__ = "MIT"

# Activate logger
from ast import arg
import logging, coloredlogs

coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__.replace("__", ""))


logger.debug("Importing dependencies...")

# Import dependencies
import simulate
from simulate.make_code import (
    make_qc_parity_check_matrix,
    make_random_ldpc_parity_check_matrix_with_identity,
    make_regular_ldpc_parity_check_matrix,
    make_regular_ldpc_parity_check_matrix_identity,
)
from simulate.utils import CommandsBase, make_random_state
from simulate.decode import (
    simulate_frame_error_rate,
    simulate_frame_error_rate_rust,
    ErrorsProvider,
)
import simulate.distance_spectrum
from simulate.hqc import simulate_hqc_idealized_oracle
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
        error_group.add_argument(
            "--threads",
            action="store",
            type=int,
            help="Number of threads to run decoders on",
            default=4,
        )

    def command_hqc_simulate(self, args: argparse.Namespace):
        rng = make_random_state(args.seed)
        simulate_hqc_idealized_oracle(rng)

    def command_test_rust_package(self, args: argparse.Namespace):
        logger.info(
            "Testing a regular (3,6+1) ldpc code with a parity check matrix of the form: H_r*k|I_r*r"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        error_rate = args.error_rate
        threads = args.threads
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

        successes = simulate_frame_error_rate_rust(H, error_rate, runs, rng, threads)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_regular_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a regular (3,6) ldpc code with a parity check matrix of the form: H_r*k"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        errors_provider = ErrorsProvider(args.error_rate, args.error_file, rng)
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
        errors_provider = ErrorsProvider(args.error_rate, args.error_file, rng)
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

        successes = simulate_frame_error_rate(H, errors_provider, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_qc_ldpc_code(self, args: argparse.Namespace):
        logger.info(
            "Testing a quasi cyclic ldpc code with a parity check matrix of the form: [H_0|H_1|I]"
        )
        rng = make_random_state(args.seed)
        runs = args.runs
        errors_provider = ErrorsProvider(args.error_rate, args.error_file, rng)
        k = 1000  # 17669
        num_blocks = 2
        r = int(k / num_blocks)
        # n = k + r
        H = make_qc_parity_check_matrix(
            block_len=r, column_weight=3, num_blocks=num_blocks, rng=rng
        )
        logger.debug("Constructed parity check matrix:\n" + str(H))

        successes = simulate_frame_error_rate(H, errors_provider, runs, rng)
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
        errors_provider = ErrorsProvider(args.error_rate, args.error_file, rng)
        runs = args.runs
        H = rep_code(n)

        successes = simulate_frame_error_rate(H, errors_provider, runs, rng)
        logger.info(f"Success ratio {successes}/{runs}={successes/runs}")

    def command_test(self, args: argparse.Namespace):
        """This command runs all discovered doctests in the simulate package"""
        self.command_test_xml(args, xml=False)

    def command_test_xml(self, args: argparse.Namespace, xml=True):
        """
        This command runs all discovered doctests in the simulate package.

        Optionally with xml format output.
        """
        import unittest
        import doctest

        if xml:
            import xmlrunner

        suite = unittest.TestSuite()
        suite.addTest(doctest.DocTestSuite(simulate.utils))
        suite.addTest(doctest.DocTestSuite(simulate.make_code))
        suite.addTest(doctest.DocTestSuite(simulate.decode))
        suite.addTest(doctest.DocTestSuite(simulate.distance_spectrum))
        suite.addTest(doctest.DocTestSuite(simulate.hqc))

        logger.info("Disabling further logging output")
        logging.disable(logging.CRITICAL)
        results = None
        if xml:
            with open("report.xml", "wb") as output:
                results = xmlrunner.XMLTestRunner(
                    output=output,
                    failfast=False,
                    buffer=False,
                ).run(suite)
        else:
            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 0)
            results = runner.run(suite)

        if results.wasSuccessful():
            exit(0)
        else:
            exit(1)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    cmds = Commands()

    args = cmds.parse_arguments()

    logger.info("Python simulator started")

    cmds.run(args)
