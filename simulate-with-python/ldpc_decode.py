import coloredlogs
import numpy as np
from simulate_rs import DecoderNTRU

coloredlogs.install(level="DEBUG", logger=None)

# new logger for this module
import logging

logger = logging.getLogger(__name__.replace("__", ""))


def process_file(filename, n):
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


# number of coefficients of f
p = 4
# weight of f
w = 2

# determine the prior distribution of coefficients of f
f_zero_prob = (p - w) / p
f_one_prob = (1 - f_zero_prob) / 2
secret_variables = []
for _ in range(p):
    secret_variables.append([f_one_prob, f_zero_prob, f_one_prob])

# read posterior distribution of check variables
filename = "to_be_decoded.txt"
H, check_variables = process_file(filename, p)

# convert to numpy arrays for Rust be able to work on the arrays
secret_variables = np.array(secret_variables, dtype=np.float32)
check_variables = np.array(check_variables, dtype=np.float32)

decoder = DecoderNTRU(H.astype("int8"), 7)
s_decoded = decoder.min_sum(secret_variables, check_variables)

print(f"Decoded key: {s_decoded[:p]}\nCheck variables: {s_decoded[p:]}")