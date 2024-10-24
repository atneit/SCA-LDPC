import itertools as it
import os
import random
from collections import defaultdict
from math import inf, log, prod

import numpy as np
from simulate.adaptive_tree_coding import (
    sample_coef_with_adaptive_coding,
    tree_from_array,
    tree_from_coding,
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

def extend_distribution(distr, target_b):
    l = len(distr)
    B = (l - 1) // 2
    res = [0.0] * (target_b * 2 + 1)
    for i, pr in enumerate(distr):
        if pr > 0:
            x = i - B
            new_idx = x + target_b
            res[new_idx] = pr
    return res


def parse_file(file_path):
    keys = []
    collisions = []

    with open(file_path, "r") as f:
        current_key = []
        in_key_section = False
        num_collisions = 0
        collision_info = []

        for line in f:
            line = line.strip()

            # Look for "pq_counter:"
            if line.startswith("pq_counter:"):
                # Reset variables for new entry
                current_key = []
                in_key_section = False
                num_collisions = 0
                collision_info = []

            # Look for "The private key is:"
            elif line == "The private key is:":
                in_key_section = True
                continue  # Skip to next line, where the key starts

            # If we are in the key section, capture the key data
            elif in_key_section:
                if line:  # If the line contains key data
                    # Remove trailing comma and split the key values
                    current_key = [int(x) for x in line.rstrip(",").split(",")]
                    in_key_section = False  # We are done with key section

            # Look for "Collisions detected:"
            elif line.startswith("Collisions detected:"):
                num_collisions = int(line.split(":")[1])

            # Capture collision index and value
            elif line.startswith("collision_index"):
                index_value = line.split(",")
                collision_index = int(index_value[0].split(":")[1])
                collision_value = int(index_value[1].split(":")[1])
                collision_info.append((collision_index, collision_value))

                # If only one collision is detected, save the key and collision info
                if num_collisions == 1:
                    keys.append(current_key)
                    collisions.append(collision_info)

    return keys, collisions


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
# SECOND: choose "good" coding tree for check variables
# THIRD: choose "good" list of check variables
# here we can choose check variables with different parameters, e.g. ask a few with
# weight 2, then a few with weight 4
attack_params = []
random.seed(0)

sum_weight = 2
u_list = []
# naive idea: generate random indices for the first row, then just shift it; repeat a few times
u_base = sorted(random.sample(range(n), sum_weight))
for offset in range(n):
    u_list.append(list((u_val + offset) % n for u_val in u_base))
coding_tree = tree_from_array([(True, 1), (False, -1), (True, 2), None, (False, -2)])
attack_params.append((u_list, coding_tree, sum_weight))
sum_weight = 4
u_list = []
# naive idea: generate random indices for the first row, then just shift it; repeat a few times
u_base = sorted(random.sample(range(n), sum_weight))
for offset in range(n):
    u_list.append(list((u_val + offset) % n for u_val in u_base))
coding_tree = tree_from_array(
    [
        (True, 1),
        (False, -1),
        (True, 2),
        None,
        (False, -2),
        None,
        (True, 3),
        None,
        None,
        None,
        (False, -3),
    ]
)
# can also create trees from the table (same result):
coding_tree = tree_from_coding(
    [
        (0, 1, 1, 1),
        (0, 1, 1, 1),
        (0, 1, 1, 0),
        (0, 1, 0),
        (0, 0),
        (1, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 1),
    ]
)
attack_params.append(
    (
        u_list,
        coding_tree,
        sum_weight,
    )
)
patterns = [(0, 1, 1), (0, 1, 0), (0, 0), (1, 0), (1, 1)]
coding_tree = tree_from_coding(patterns)

# check that among our check variables there are no cycles of length 4
total_u_list = sum((params[0] for params in attack_params), [])
is_bad_list = has_length_4_cycle(total_u_list)
assert (
    not is_bad_list
), "Check variables have a cycle of length 4, try different random.seed()"
max_sum_weight = max(params[2] for params in attack_params)


pr_oracle = SimpleOracle(0.95)  # Bernoulli noise
s_distr = secret_distr()
beta_distr = list(sum_secret_distr(s_distr, i + 1) for i in range(sum_weight))
# for i, distr in enumerate(beta_distr):
#     print(f"weight = {i+1}, entropy = {float(entropy(distr)):.3f}")
#     print(", ".join(f"{float(x):.4f}" for x in distr.values()))
s_distr_flat = list((s_distr[-1], s_distr[0], s_distr[1]))
s_prior = list(s_distr_flat for _ in range(n))

# print("Coding tree gives (no more than) the following information per check variable")
# for i, distr in enumerate(beta_distr):
#     if i % 2 == 1:
#         continue
#     info, avg_length = information_for_coding_tree(
#         pr_oracle, secret_range, coding_tree, distr, i + 1
#     )
#     print(f"{i=}, {info=}, {avg_length=}")

# print("====================")

TEST_KEYS = 100

keys, collisions = parse_file("private_key_and_collision_info.bin")
differences_arr = []
oracle_calls_arr = []
key_num = 0
for key, collision_info in zip(keys, collisions):
    if key_num >= TEST_KEYS:
        break
    f = to_centered(key)
    col_idx = collision_info[0][0]
    # reset oracle count for secret key
    pr_oracle.oracle_calls = 0

    alpha_idxs = []
    cond_distr = []
    for u_list, coding_tree, sum_weight in attack_params:
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
            if sum_weight == max_sum_weight:
                cond_distr.append(distr)
            else:
                cond_distr.append(extend_distribution(distr, max_sum_weight))

    s_decoded = ldpc_decode(alpha_idxs, cond_distr, 11, s_prior, max_sum_weight)
    fprime = s_decoded[:n]
    differences = sum(f[i] != fprime[i] for i in range(len(f)))
    differences_arr.append(differences)
    oracle_calls_arr.append(pr_oracle.oracle_calls)
    key_num += 1


print(f"Average number of incorrect coefficients = {np.average(differences_arr)}")
print(f"Average number of oracle calls = {np.average(oracle_calls_arr)}")