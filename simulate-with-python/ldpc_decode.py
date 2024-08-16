import os.path
import sys

import coloredlogs
import numpy as np
from simulate_rs import DecoderNTRU761W2

coloredlogs.install(level="DEBUG", logger=None)

# new logger for this module
import logging

logger = logging.getLogger(__name__.replace("__", ""))


def process_file(filename, n):
    if not os.path.isfile(filename):
        print("File does not exist")
        return None, None

    with open(filename, "r") as file:
        lines = file.readlines()

    index_lines = []
    probability_lists = []

    # read lines in blocks of 2
    for i in range(0, len(lines), 2):
        indices = list(map(int, lines[i].strip().split(", ")))
        probabilities = list(map(float, lines[i + 1].strip().split(",")))

        index_lines.append(indices)
        probability_lists.append(probabilities)

    # Determine the size of the matrix
    num_rows = len(index_lines)
    # Create the matrix with the appropriate size
    matrix = np.zeros((num_rows, n + num_rows), dtype=int)

    # Fill in the ones based on indices and append the negative identity matrix
    for i, indices in enumerate(index_lines):
        for index in indices:
            matrix[i, index] = 1
        matrix[i, n + i] = -1

    return matrix, probability_lists


argv = sys.argv
if len(argv) < 3:
    print("Usage: <program> <prob_file> <out_file>")
    exit()

# number of coefficients of f
p = 761
# weight of f
w = 286

# determine the prior distribution of coefficients of f
f_zero_prob = (p - w) / p
f_one_prob = (1 - f_zero_prob) / 2
secret_variables = []
for _ in range(p):
    secret_variables.append([f_one_prob, f_zero_prob, f_one_prob])

# read posterior distribution of check variables
filename = argv[1]
H, check_variables = process_file(filename, p)
if H is None or check_variables is None:
    exit()

# convert to numpy arrays for Rust be able to work on the arrays
secret_variables = np.array(secret_variables, dtype=np.float32)
check_variables = np.array(check_variables, dtype=np.float32)

decoder = DecoderNTRU761W2(H.astype("int8"), 7)
s_decoded = decoder.min_sum(secret_variables, check_variables)

with open(argv[2], "wt") as f:
    print(s_decoded[:p], file=f)
    print(s_decoded[p:], file=f)