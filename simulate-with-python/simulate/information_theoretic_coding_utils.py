from collections import defaultdict
from math import log

from simulate.adaptive_tree_coding import traverse_all_paths_for_value
from simulate.max_likelihood import s_distribution_from_hard_y_adaptive


def compute_entropy(distr):
    entropy = 0
    for p in distr:
        if p == 0:
            continue
        entropy += p * log(p, 2)
    return -entropy


def compute_probability_of_all_y_adaptive(pr_oracle, distr, coding_tree):
    pr_of_y = defaultdict(float)
    for s in distr.keys():
        for y, pr in traverse_all_paths_for_value(coding_tree, pr_oracle, s):
            pr_of_y[y] += pr * distr[s]
    return pr_of_y


def compute_conditional_distributions_adaptive(
    pr_oracle, secret_range_func, coding_tree, sum_weight, distrib_secret
):
    pr_of_y = compute_probability_of_all_y_adaptive(
        pr_oracle, distrib_secret, coding_tree
    )
    cond_distributions = {}
    for y in pr_of_y.keys():
        cond_distr = s_distribution_from_hard_y_adaptive(
            y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
        )
        cond_distributions[y] = cond_distr
    return cond_distributions, pr_of_y


def information_for_coding_tree(
    pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    cond_distributions, pr_of_y = compute_conditional_distributions_adaptive(
        pr_oracle, secret_range_func, coding_tree, sum_weight, distrib_secret
    )
    e = 0
    for y, cond_distr in cond_distributions.items():
        entropy = compute_entropy(cond_distr)
        e += entropy * pr_of_y[y]
    info = compute_entropy(distrib_secret.values()) - e
    avg_length = 0
    for y, pr in pr_of_y.items():
        avg_length += len(y) * pr
    return info, avg_length


if __name__ == "__main__":
    import itertools as it
    from math import prod

    from adaptive_tree_coding import tree_from_array
    from max_likelihood import SimpleOracle

    def secret_distr(p, w):
        f_zero_prob = (p - w) / p
        f_one_prob = (1 - f_zero_prob) / 2
        return {-1: f_one_prob, 0: f_zero_prob, 1: f_one_prob}

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

    pr_oracle = SimpleOracle(0.96)
    coding_tree = tree_from_array(
        [(True, 1), (False, -1), (True, 2), None, (False, -2)]
    )
    sum_weight = 2
    distr = sum_secret_distr(secret_distr(761, 246), sum_weight)
    pr_of_y = compute_probability_of_all_y_adaptive(pr_oracle, distr, coding_tree)
    print(pr_of_y)
    cond_distributions, pr_of_y = compute_conditional_distributions_adaptive(
        pr_oracle, secret_range, coding_tree, 2, distr
    )

    print(cond_distributions)
    info, avg_length = information_for_coding_tree(
        pr_oracle, secret_range, coding_tree, distr, sum_weight
    )
    print(f"{info=}, {avg_length=}")
