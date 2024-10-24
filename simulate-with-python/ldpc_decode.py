import os.path
import sys

import coloredlogs
import numpy as np
from simulate_rs import DecoderNTRU761W2, DecoderNTRU761W4

coloredlogs.install(level="DEBUG", logger=None)

# new logger for this module
import logging

logger = logging.getLogger(__name__.replace("__", ""))

MOVE_SINGLE_CHECKS_TO_APRIOR = True


def process_file(filename, n, check_weight):
    if not os.path.isfile(filename):
        print("File does not exist")
        return None, None

    with open(filename, "r") as file:
        lines = file.readlines()

    index_lines = []
    probability_lists = []

    single_check_idxs = []
    single_check_distr = []

    # read lines in blocks of 2
    for i in range(0, len(lines), 2):
        indices = list(map(int, lines[i].strip().split(", ")))
        probabilities = list(map(float, lines[i + 1].strip().split(",")))

        # support the case where extra probabilities are not printed
        if len(probabilities) == len(indices) * 2 + 1 and len(indices) < check_weight:
            offset = check_weight - len(indices)
            probabilities = [0.0] * offset + probabilities + [0.0] * offset

        if MOVE_SINGLE_CHECKS_TO_APRIOR and len(indices) == 1:
            single_check_idxs.append(indices[0])
            single_check_distr.append(probabilities)
        else:
            index_lines.append(indices)
            probability_lists.append(probabilities)

    # Determine the number of parity checks
    num_rows = len(index_lines)
    # Create the matrix with the appropriate size
    matrix = np.zeros((num_rows, n + num_rows), dtype=int)

    # Fill in the ones based on indices and append the negative identity matrix
    for i, indices in enumerate(index_lines):
        for index in indices:
            matrix[i, index] = 1
        matrix[i, n + i] = -1

    return matrix, probability_lists, single_check_idxs, single_check_distr


argv = sys.argv
if len(argv) < 3:
    print("Usage: <program> <prob_file> <out_file> [<LDPC_iterations>]")
    exit()

iterations = 5
if len(argv) >= 4:
    iterations = int(argv[3])
# number of coefficients of f
p = 761
# weight of f
w = 286
check_weight = 4

# read posterior distribution of check variables
filename = argv[1]
H, check_variables, single_check_idxs, single_check_distr = process_file(
    filename, p, check_weight
)
if H is None or check_variables is None:
    exit()
row_counts = np.count_nonzero(H, axis=1)
max_row_weight = np.max(row_counts)
col_counts = np.count_nonzero(H, axis=0)
max_col_weight = np.max(col_counts)

print(len(H), len(H[0]))
print(max_row_weight, max_col_weight)
print(f"min={np.min(col_counts)}; variance = {np.var(col_counts)}")

# determine the prior distribution of coefficients of f
f_zero_prob = (p - w) / p
f_one_prob = (1 - f_zero_prob) / 2
secret_variables = []

single_checks = sorted(zip(single_check_idxs, single_check_distr))
single_checks_idx = 0
for i in range(p):
    if (
        single_checks_idx < len(single_checks)
        and single_checks[single_checks_idx][0] == i
    ):
        distr = single_checks[single_checks_idx][1]
        l = len(distr)
        secret_variables.append(distr[l // 2 - 1 : l // 2 + 2])
        single_checks_idx += 1
    else:
        secret_variables.append([f_one_prob, f_zero_prob, f_one_prob])

# convert to numpy arrays for Rust be able to work on the arrays
secret_variables = np.array(secret_variables, dtype=np.float32)
check_variables = np.array(check_variables, dtype=np.float32)

# print("Creating decoder")
if check_weight == 2:
    decoder = DecoderNTRU761W2(
        H.astype("int8"), max_col_weight, max_row_weight, iterations
    )
elif check_weight == 4:
    decoder = DecoderNTRU761W4(
        H.astype("int8"), max_col_weight, max_row_weight, iterations
    )
else:
    raise ValueError("Not supported check weight")
# print("Decoder created, computing min_sum")
s_decoded = decoder.min_sum(secret_variables, check_variables)
# print("Done")

with open(argv[2], "wt") as f:
    print(s_decoded[:p], file=f)
    print(s_decoded[p:], file=f)