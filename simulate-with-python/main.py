#!/bin/env python
"""
Module Docstring
"""

__author__ = "Alexander Nilsson; Denis Nabokov; Qian Guo"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
from logzero import logger
from simulate.utils import CommandsBase

class Commands(CommandsBase):
    """
    All possible commands for this simulator to run
    """

    PREFIX="command_"
    
    def command_official_example(self, args):
        """A simple command to test the functionality of the ldpc package"""
        logger.info("official example")
        import numpy as np
        from ldpc.codes import rep_code
        from ldpc import bp_decoder

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



def main(args):
    """ Main entry point of the app """
    logger.info("Python simulator started")

    cmds = Commands()
    cmds.run(args)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # What command should execute
    parser.add_argument("command", help="What command should execute?")

    args = parser.parse_args()
    main(args)