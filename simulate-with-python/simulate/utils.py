"""
Module utils
"""

import argparse
import sys
from logzero import logger

class CommandsBase:
    """ Base class for running commands
    """

    def __init__(self):

        logger.debug("Setting up arguments parser...")
        self._parser = argparse.ArgumentParser()

        # What command should execute
        possible_commands = [command[len(self.PREFIX):] for command in dir(self) if self.PREFIX in command]
        self._parser.add_argument("command", help="What command should execute? Possible values: " + str(possible_commands))

        logger.debug("Adding custom arguments to parser...")
        self.setup_arguments(self._parser)
        

    def setup_arguments(self, parser):
        """Add custom arguments to parser here for commands"""
        pass

    def parse_arguments(self, args: list=None):
        """
        Parse command line arguments.

        Parameters:
            args (None | list of strings) - If None, use sys.argv
        """
        logger.debug("Parsing given command line arguments...")
        return self._parser.parse_args(args)


    def run(self, args: argparse.Namespace):
        command = str(args.command)
        func = getattr(self, self.PREFIX + command, None)
        if func:
            logger.info(f"Executing \"{command}\" with arguments: {args}")
            return func(args)
        else:
            logger.error("Bad command given")
            sys.exit(1)