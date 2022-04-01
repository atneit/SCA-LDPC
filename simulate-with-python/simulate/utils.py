"""
Module utils
"""

import argparse
import sys
import logzero
from logzero import logger
import numpy as np
import numbers


def pretty_string_matrix(matrix: list):
    return "[" + "\n ".join([str(row) for row in matrix]) + "]"


def make_random_state(seed) -> np.random.RandomState:
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


class CommandsBase:
    """Base class for running commands"""

    def __init__(self):

        logger.debug("Setting up arguments parser...")
        self._parser = argparse.ArgumentParser()

        # What command should execute
        possible_commands = [
            command[len(self.PREFIX) :]
            for command in dir(self)
            if self.PREFIX in command
        ]
        self._parser.add_argument(
            "command",
            help="What command should execute? Possible values: "
            + str(possible_commands),
        )
        self._parser.add_argument("--verbose", "-v", action="count", default=0)

        logger.debug("Adding custom arguments to parser...")
        self.setup_arguments(self._parser)

    def setup_arguments(self, parser):
        """Add custom arguments to parser here for commands"""
        pass

    def parse_arguments(self, args: list = None):
        """
        Parse command line arguments.

        Parameters:
            args (None | list of strings) - If None, use sys.argv
        """
        logger.debug("Parsing given command line arguments...")
        parsed = self._parser.parse_args(args)
        if not parsed.verbose:
            logger.debug("Turning off debug messages from this point onward")
            logzero.loglevel(logzero.INFO)
        return parsed

    def run(self, args: argparse.Namespace):
        command = str(args.command)
        func = getattr(self, self.PREFIX + command, None)
        if func:
            logger.info(f'Executing "{command}" with arguments: {args}')
            return func(args)
        else:
            logger.error("Bad command given: " + command)
            sys.exit(1)
