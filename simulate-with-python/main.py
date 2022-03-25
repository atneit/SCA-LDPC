#!/bin/env python
"""
Main entry point for the python LDPC code simulator.
"""

__author__ = "Alexander Nilsson; Denis Nabokov; Qian Guo"
__version__ = "0.1.0"
__license__ = "MIT"

from logzero import logger
logger.debug("Importing dependencies...")

import argparse
from simulate.utils import CommandsBase
import numpy as np
from ldpc.codes import rep_code
from ldpc import bp_decoder

logger.debug("Importing dependencies... Done")

class Commands(CommandsBase):
    """
    This class defines all possible commands for this simulator to run
    """

    PREFIX="command_"

    def setup_arguments(self, parser: argparse.ArgumentParser):
        pass

    def command_regular_ldpc_code(self, args: argparse.Namespace):
        pass
    
    def command_official_example(self, args: argparse.Namespace):
        """A simple command to test the functionality of the ldpc package"""
        logger.info("official example")


        n=13
        error_rate=0.3
        runs=5
        H=rep_code(n)

        #BP decoder class. Make sure this is defined outside the loop
        bpd=bp_decoder(H,error_rate=error_rate,max_iter=n,bp_method="product_sum")
        error=np.zeros(n).astype(int) #error vector

        for _ in range(runs):
            for i in range(n):
                if np.random.random()<error_rate:
                    error[i]=1
                else: error[i]=0
            syndrome=H@error %2 #calculates the error syndrome
            print(f"Error: {error}")
            print(f"Syndrome: {syndrome}")
            decoding=bpd.decode(syndrome)
            print(f"Decoding: {error}\n")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    cmds = Commands()
    
    args = cmds.parse_arguments()

    logger.info("Python simulator started")

    cmds.run(args)