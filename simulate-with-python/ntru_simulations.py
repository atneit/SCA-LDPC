import itertools as it
import os
import random
from collections import defaultdict
from math import inf, log, prod

import numpy as np
from simulate.adaptive_tree_coding import (
    sample_coef_with_adaptive_coding,
    tree_from_array,
)
from simulate.information_theoretic_coding_utils import information_for_coding_tree
from simulate.max_likelihood import (
    FalsePositiveNegativePositionalOracle,
    SimpleOracle,
    s_distribution_from_hard_y,
    s_distribution_from_hard_y_adaptive,
)
from simulate_rs import DecoderNTRUW2, DecoderNTRUW4, DecoderNTRUW6

q = 2048
n = 509


def entropy(distr):
    if type(distr) is list:
        return -sum(p * log(p, 2) for p in distr if p != 0)
    else:
        return -sum(p * log(p, 2) for p in distr.values() if p != 0)


def secret_distr():
    return {-1: 85 / 256, 0: 86 / 256, 1: 85 / 256}


def sum_secret_distr(distr, weight):
    B = (len(distr) - 1) // 2
    d = defaultdict(float)
    for values in it.product(range(-B, B + 1), repeat=weight):
        d[sum(values)] += prod(distr[val] for val in values)
    return d


def secret_range(sum_weight):
    return range(-sum_weight, sum_weight + 1)


def secret_range_len(sum_weight):
    return 2 * sum_weight + 1


def coding_from_patterns(pattern, sum_weight=1):
    B = sum_weight
    if len(pattern) != (2 * B + 1):
        raise ValueError("len of pattern doesn't match sum weight")
    if type(pattern[0]) is tuple:
        return {s: p_val for s, p_val in zip(range(-B, B + 1), pattern)}
    else:
        return {s: (p_val,) for s, p_val in zip(range(-B, B + 1), pattern)}


def compute_alpha(u, i, n):
    return (i - u) % n


def to_centered(f):
    ret = [0] * len(f)
    for i in range(len(f)):
        x = int(f[i])
        if x >= q // 2:
            x -= q
        ret[i] = x
    return ret


def has_length_4_cycle(res_idxs):
    # Get the number of check variables
    num_check_vars = len(res_idxs)

    # Iterate over all pairs of check variables (i, j)
    for i in range(num_check_vars):
        for j in range(i + 1, num_check_vars):
            # Find the intersection of the secret variables for check i and check j
            common_vars = set(res_idxs[i]).intersection(res_idxs[j])

            # Check if there are 2 or more common secret variables
            if len(common_vars) >= 2:
                print(i, j, res_idxs[i], res_idxs[j])
                return True  # A length-4 cycle is found

    return False  # No length-4 cycles found


def ldpc_decode(idxs, distrs, iterations, prior_secret_variables, sum_weight):
    num_rows = len(idxs)
    matrix = np.zeros((num_rows, n + num_rows), dtype=int)

    for i, indices in enumerate(idxs):
        # add 1 for even indices, -1 for odd
        for j, index in enumerate(indices):
            matrix[i, index] = (-1) ** (j % 2)
        matrix[i, n + i] = -1
    row_counts = np.count_nonzero(matrix, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(matrix, axis=0)
    max_col_weight = np.max(col_counts)

    secret_variables = np.array(prior_secret_variables, dtype=np.float32)
    check_variables = np.array(distrs, dtype=np.float32)

    if sum_weight == 2:
        decoder_class = DecoderNTRUW2
    elif sum_weight == 4:
        decoder_class = DecoderNTRUW4
    elif sum_weight == 6:
        decoder_class = DecoderNTRUW6
    else:
        raise ValueError(f"{sum_weight} weight for check variable is not supported")
    decoder = decoder_class(
        matrix.astype("int8"), max_col_weight, max_row_weight, iterations
    )
    s_decoded = decoder.min_sum(secret_variables, check_variables)
    return s_decoded


# FIRST: choose "good" weight of check variables
sum_weight = 2
# SECOND: choose "good" coding tree for check variables
coding_tree = tree_from_array([(True, 1), (False, -1), (True, 2), None, (False, -2)])
# THIRD: choose "good" list of check variables
u_list = []
random.seed(4)
# naive idea: generate random indices for the first row, then just shift it; repeat a few times
for _ in range(2):
    u_base = sorted(random.sample(range(n), sum_weight))
    for offset in range(n):
        u_list.append(list((u_val + offset) % n for u_val in u_base))
# check that among our check variables there are no cycles of length 4
is_bad_list = has_length_4_cycle(u_list)
assert not is_bad_list


pr_oracle = SimpleOracle(0.99)  # Bernoulli noise
# pr_oracle = SimpleOracle(1)  # Bernoulli noise
s_distr = secret_distr()
beta_distr = list(sum_secret_distr(s_distr, i + 1) for i in range(sum_weight))
for i, distr in enumerate(beta_distr):
    print(f"weight = {i+1}, entropy = {float(entropy(distr)):.3f}")
    print(", ".join(f"{float(x):.4f}" for x in distr.values()))
s_distr_flat = list((s_distr[-1], s_distr[0], s_distr[1]))
s_prior = list(s_distr_flat for _ in range(n))

print("Coding tree gives (no more than) the following information per check variable")
for i, distr in enumerate(beta_distr):
    if i % 2 == 1:
        continue
    info, avg_length = information_for_coding_tree(
        pr_oracle, secret_range, coding_tree, distr, i + 1
    )
    print(f"{i=}, {info=}, {avg_length=}")

print("====================")

# TODO: parse file and get data from it
# fmt: off
f = [2047,1,0,1,1,2047,1,0,1,1,1,2047,1,1,0,1,2047,0,0,2047,1,2047,1,2047,2047,1,0,1,0,2047,0,1,0,1,2047,1,1,0,1,0,0,2047,1,0,0,0,1,0,0,1,1,0,0,2047,2047,0,0,1,2047,1,1,2047,0,0,2047,2047,0,2047,1,0,2047,0,0,1,0,0,2047,1,0,1,0,2047,2047,2047,2047,1,2047,1,0,1,1,2047,1,1,0,1,0,1,2047,0,2047,1,0,1,2047,0,2047,2047,0,1,0,0,1,2047,1,2047,0,2047,0,0,0,1,1,1,0,1,2047,2047,0,0,1,0,2047,1,1,2047,2047,1,1,0,0,0,1,1,2047,0,1,1,2047,0,1,2047,2047,1,2047,0,0,2047,1,2047,0,2047,1,0,0,2047,1,0,0,1,0,2047,0,2047,1,1,2047,1,1,0,0,1,0,1,1,2047,0,2047,2047,0,1,1,2047,0,2047,0,2047,1,0,0,1,0,1,2047,0,0,0,1,2047,0,1,2047,1,1,1,0,0,2047,0,2047,2047,0,2047,0,2047,2047,0,2047,1,1,2047,0,0,1,0,2047,1,1,1,1,2047,1,1,0,2047,0,2047,1,2047,0,0,0,0,1,0,0,1,0,2047,1,2047,1,1,2047,1,0,0,2047,2047,2047,0,1,0,2047,0,2047,1,0,1,2047,2047,0,1,0,2047,2047,2047,2047,2047,1,0,2047,1,2047,2047,0,2047,1,1,2047,1,1,2047,1,1,2047,1,2047,2047,0,0,1,1,1,1,0,2047,2047,2047,2047,1,0,2047,1,0,1,0,1,2047,1,2047,0,0,1,0,1,1,2047,0,0,2047,2047,0,2047,2047,1,1,2047,1,0,1,2047,1,1,2047,0,1,2047,1,1,1,0,2047,2047,1,0,2047,2047,2047,2047,2047,0,1,0,1,0,1,0,1,1,2047,1,2047,0,1,2047,0,1,0,2047,1,0,2047,1,0,0,0,1,2047,0,2047,0,1,0,2047,2047,2047,2047,2047,1,0,1,2047,0,0,2047,0,0,2047,1,0,1,2047,2047,0,1,2047,0,1,2047,1,0,0,1,1,0,2047,1,0,2047,2047,1,1,2047,2047,0,2047,2047,2047,0,0,2047,2047,0,0,1,1,0,1,2047,2047,0,1,1,1,2047,2047,2047,0,1,0,1,0,0,0,2047,1,2047,0,0,1,0,0,2047,0,2047,0,2047,1,2047,2047,2047,2047,1,1,2047,2047,0,2047,2047,2047,1,2047,1,1,0,1,0,0,]
# fmt: on
f = to_centered(f)
col_idx = 427
# reset oracle count for secret key
pr_oracle.oracle_calls = 0

alpha_idxs = []
cond_distr = []
for u in u_list:
    alpha = list(compute_alpha(u_val, col_idx, n) for u_val in u)
    beta_u = 0
    for i, idx in enumerate(alpha):
        # for even i we add coefficient of f, otherwise we subtract it
        f_coef = f[idx]
        if (i % 2) == 0:
            beta_u += f_coef
        else:
            beta_u -= f_coef
    y = sample_coef_with_adaptive_coding(pr_oracle, beta_u, coding_tree)
    distr = s_distribution_from_hard_y_adaptive(
        y,
        pr_oracle,
        secret_range,
        coding_tree,
        beta_distr[sum_weight - 1],
        sum_weight,
    )
    alpha_idxs.append(alpha)
    cond_distr.append(distr)


# print(u_list[0])
# for u_val in u_list[0]:
#     print(u_val, "->", compute_alpha(u_val, col_idx, n))
#     print(f[compute_alpha(u_val, col_idx, n)])
# print(cond_distr[0])

s_decoded = ldpc_decode(alpha_idxs, cond_distr, 11, s_prior, sum_weight)
fprime = s_decoded[:n]
# print(f)
# print(fprime)
differences = sum(f[i] != fprime[i] for i in range(len(f)))
print(f"{differences=}")
print(f"Used {pr_oracle.oracle_calls} oracle calls")
