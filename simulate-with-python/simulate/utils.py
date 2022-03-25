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

        self.setup_arguments(self._parser)

    def setup_arguments(self, parser):
        pass

    def run(self):
        args = self._parser.parse_args()
        command = str(args.command)
        func = getattr(self, self.PREFIX + command, None)
        if func:
            logger.info(f"Executing \"{command}\" with arguments: {args}")
            return func(args)
        else:
            logger.error("Bad command given")
            sys.exit(1)