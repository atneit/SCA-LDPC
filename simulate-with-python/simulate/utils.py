"""
Module utils
"""

import sys
from logzero import logger

class CommandsBase:
    """ Base class for running commands
    """
    def run(self, args):
        command = str(args.command)
        func = getattr(self, self.PREFIX + command, None)
        if func:
            logger.info(f"Executing {command} with arguments: {args}")
            return func(args)
        else:
            logger.error("Bad command given")
            sys.exit(1)