import itertools as it
from collections import deque


# tree defines a set of checks for a variable X either X >= value if ge_flag = True
# or X <= value, otherwise
class Node:
    def __init__(self, ge_flag, value, left=None, right=None):
        self.ge_flag = ge_flag
        self.value = value
        self.left = left
        self.right = right


def recursive_patterns_scan(patterns, position, begin, end, B):
    if end - begin == 1 or position >= len(patterns[begin]):
        return None
    start_val = patterns[begin][position]
    for i in range(begin, end):
        p = patterns[i]
        if p[position] != start_val:
            # at index i we have a switch
            if start_val == 0:
                n = Node(ge_flag=True, value=(i - B))
                n.left = recursive_patterns_scan(patterns, position + 1, begin, i, B)
                n.right = recursive_patterns_scan(patterns, position + 1, i, end, B)
            else:
                n = Node(ge_flag=False, value=(i - B - 1))
                n.right = recursive_patterns_scan(patterns, position + 1, begin, i, B)
                n.left = recursive_patterns_scan(patterns, position + 1, i, end, B)
            return n
    raise NotImplementedError(
        "Should add support to the case when there is no switch in recursive_patterns_scan"
    )

# Assume that each call to oracle gives information of the form "s < x" for fixed x
def tree_from_coding(patterns):
    B = len(patterns) // 2
    assert len(patterns) == 2 * B + 1
    return recursive_patterns_scan(patterns, 0, 0, len(patterns), B)


# Returns probability for each possible path in the tree. The path is taken for <value>, i.e. we check
# going left or right depending on it
def traverse_all_paths_for_value(root, pr_oracle, value):
    d = deque([(root, tuple(), 1)])  # Start with root, empty label, and probability 1
    while len(d) != 0:
        node, label, prob = d.pop()
        if node is None:
            yield (label, prob)
            continue

        pos = (node.ge_flag, node.value)  # Node position as used in oracle
        if node.ge_flag:
            if value >= node.value:
                # Predict for value >= node.value, i.e. expected = 1
                d.append(
                    (node.right, label + (1,), prob * pr_oracle.prob_of(1, 1, pos))
                )  # Going right, output 1
                d.append(
                    (node.left, label + (0,), prob * pr_oracle.prob_of(1, 0, pos))
                )  # Going left, output 0
            else:
                # Predict for value < node.value, i.e. expected = 0
                d.append(
                    (node.left, label + (0,), prob * pr_oracle.prob_of(0, 0, pos))
                )  # Going left, output 0
                d.append(
                    (node.right, label + (1,), prob * pr_oracle.prob_of(0, 1, pos))
                )  # Going right, output 1
        else:
            if value <= node.value:
                # Predict for value <= node.value, i.e. expected = 1
                d.append(
                    (node.right, label + (1,), prob * pr_oracle.prob_of(1, 1, pos))
                )  # Going right, output 1
                d.append(
                    (node.left, label + (0,), prob * pr_oracle.prob_of(1, 0, pos))
                )  # Going left, output 0
            else:
                # Predict for value > node.value, i.e. expected = 0
                d.append(
                    (node.left, label + (0,), prob * pr_oracle.prob_of(0, 0, pos))
                )  # Going left, output 0
                d.append(
                    (node.right, label + (1,), prob * pr_oracle.prob_of(0, 1, pos))
                )  # Going right, output 1


# def get_all_coding_tree_arrays(max_depth, B):
#     # TODO: very stupid implementation
#     options = [None] + list(range(-B, B + 1))
#     l = 2**max_depth - 1
#     for arr in it.product(options, repeat=l):
#         arr = list(arr)
#         for i, val in enumerate(arr):
#             if val is None:
#                 if 2 * i + 1 < l:
#                     arr[2 * i + 1] = None
#                 if 2 * i + 2 < l:
#                     arr[2 * i + 2] = None
#         yield arr


def depth_first_traverse(root):
    d = deque([root])
    while len(d) != 0:
        node = d.pop()
        yield node.value
        if node.right is not None:
            d.append(node.right)
        if node.left is not None:
            d.append(node.left)


def recursive_tree_from_array(arr, i, n):
    if i >= n or arr[i] is None:
        return None

    ge_flag, value = arr[i]
    root = Node(ge_flag, value)

    root.left = recursive_tree_from_array(arr, 2 * i + 1, n)
    root.right = recursive_tree_from_array(arr, 2 * i + 2, n)

    return root


def tree_from_array(arr):
    return recursive_tree_from_array(arr, 0, len(arr))


def sample_coef_with_adaptive_coding(oracle, actual_coef, coding_tree):
    out = []
    node = coding_tree
    while node is not None:
        pos = (node.ge_flag, node.value)
        if node.ge_flag:
            # Check if X >= value
            if actual_coef >= node.value:
                b = oracle.predict_bit(1, pos)
            else:
                b = oracle.predict_bit(0, pos)
        else:
            # Check if X <= value
            if actual_coef <= node.value:
                b = oracle.predict_bit(1, pos)
            else:
                b = oracle.predict_bit(0, pos)
        out.append(b)
        if b == 1:
            node = node.right
        else:
            node = node.left
    return tuple(out)


if __name__ == "__main__":
    from collections import defaultdict
    from math import prod

    from max_likelihood import (
        FalsePositiveNegativePositionalOracle,
        SimpleOracle,
        s_distribution_from_hard_y_adaptive,
    )

    def secret_distr(p, w):
        f_zero_prob = (p - w) / p
        f_one_prob = (1 - f_zero_prob) / 2
        return {-1: f_one_prob, 0: f_zero_prob, 1: f_one_prob}

    def sum_secret_distr(distr, weight):
        B = (len(distr) - 1) // 2
        BSUM = B * weight
        d = defaultdict(float)
        for values in it.product(range(-B, B + 1), repeat=weight):
            d[sum(values)] += prod(distr[val] for val in values)
        return d

    def secret_range(sum_weight):
        return range(-sum_weight, sum_weight + 1)

    def secret_range_len(sum_weight):
        return 2 * sum_weight + 1

    sum_weight = 2
    s_distr = secret_distr(761, 286)
    distr = sum_secret_distr(s_distr, sum_weight)

    oracle = SimpleOracle(1)
    coding_arr = [(True, 1), (False, -1), (True, 2), None, (False, -2)]
    coding_tree = tree_from_array(coding_arr)

    print("Chosen encoding")
    for s in range(-2, 2 + 1):
        print(f"{s} \t| {sample_coef_with_adaptive_coding(oracle, s, coding_tree)}")

    dis1 = [0.03, 0.03]
    dis1_inv = [0.03, 0.03]
    dis2 = [0.01, 0.01]
    dis2_inv = [0.01, 0.01]
    oracle_accuracy = {
        (True, 1): dis1,
        (False, -1): dis1,
        (True, 2): dis2,
        (False, -2): dis2,
    }
    pr_oracle = FalsePositiveNegativePositionalOracle(oracle_accuracy)
    # example output
    y = (0, 0)
    print(
        s_distribution_from_hard_y_adaptive(
            y, pr_oracle, secret_range, coding_tree, distr, sum_weight
        )
    )
