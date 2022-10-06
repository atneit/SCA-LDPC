#!/bin/env python
"""
Main entry point for the python LDPC code simulator.
"""

__author__ = "Alexander Nilsson; Denis Nabokov; Qian Guo"
__version__ = "0.1.0"
__license__ = "MIT"


import coloredlogs
#colored logs for the root logger (applies for all imported modules as well)
coloredlogs.install(level="DEBUG", logger=None)

# new logger for this module
import logging
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
from simulate.visualize import view_hqc_oracle_accuracy, view_hqc_simulation_csv
import simulate.distance_spectrum
from simulate.hqc import simulate_hqc_idealized_oracle
from simulate.hqc_eval_oracle import hqc_eval_oracle
from ldpc import bp_decoder
from ldpc.codes import rep_code
import numpy as np
import argparse
import sys
from os.path import exists


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
            "--decode-every",
            action="store",
            type=int,
            default=500,
            help="Try to decode the simulation results every 'decode-every' time we add a data point.",
        )
        parser.add_argument(
            "--key-file",
            action="store",
            type=str,
            help="Serialzied key location. If it does not exist it will be created and filled with a random key (independent of --seed argument).",
        )
        parser.add_argument(
            "--csv-output",
            action="store",
            type=str,
            help="Write decoding stats to the specified csv file",
        )
        parser.add_argument(
            "--code-weight",
            action="store",
            type=int,
            default=20,
            help="Use this value as column weight for constructed LDPC code in HQC simulations",
        )
        parser.add_argument(
            "--label",
            action="store",
            type=str,
            help="Add this label to csv output to distinguish multiple runs",
        )
        error_group = parser.add_mutually_exclusive_group(required=False)
        error_group.add_argument(
            "--error-rate",
            action="store",
            type=float,
            default=0.00,
            help=("The error rate of the simulated binary symmetric channel. 'NaN' is special"
            " in that it guarantees no errors even for HQC simulation."),
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
        (_, tracking) = simulate_hqc_idealized_oracle(rng, args.decode_every, args.code_weight, args.key_file, args.error_rate)
        df = tracking.decoder_stats_data_frame(label=args.label)
        logger.info(f"Stats: \n{df.to_string(index=False)}")
        if args.csv_output:
            header = True
            mode = 'w'
            if exists(args.csv_output):
                header = False
                mode = 'a'
            df.to_csv(args.csv_output, mode=mode, index=False, header=header)
        
    def command_hqc_eval_oracle(self, args: argparse.Namespace):
        rng = make_random_state(args.seed)
        hqc_eval_oracle(rng)
        
    def command_view_hqc_oracle_accuracy(self, args: argparse.Namespace):
        view_hqc_oracle_accuracy()

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

    def command_view_hqc_simulation_csv(self, args: argparse.Namespace):
        view_hqc_simulation_csv(args.csv_output)

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
        suite.addTest(doctest.DocTestSuite(simulate.hqc_eval_oracle))

        logger.info("Starting tests and disabling further logging output")
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

    cmds.run(args)
