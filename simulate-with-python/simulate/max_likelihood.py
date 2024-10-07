import itertools as it
import random


class BaseOracle:
    def __init__(self):
        self.oracle_calls = 0

    def prob_of(self, expected, actual, pos):
        raise NotImplementedError()


class SimpleOracle(BaseOracle):
    # p: accuracy of oracle
    def __init__(self, p):
        super().__init__()
        self.p = p

    def prob_of(self, expected, actual, pos):
        if actual == expected:
            return self.p
        else:
            return 1 - self.p

    def predict_bit(self, actual_bit, pos):
        self.oracle_calls += 1
        rnd = random.random()
        if rnd < self.p:
            return actual_bit
        else:
            return 1 - actual_bit


class FalsePositiveNegativePositionalOracle(BaseOracle):
    # as input p_positional should be array of tuples, where tuple i contains a pair of
    # probability of false positive and false negative, resp., for pos == i.
    # Alternatively, p_positional could be an dictionary, with values being tuples as before
    def __init__(self, p_positional):
        super().__init__()
        self.p_positional = p_positional

    def prob_of(self, expected, actual, pos):
        pr_fp, pr_fn = self.p_positional[pos]
        if expected == 0:
            if actual == 1:
                return pr_fp
            else:
                return 1 - pr_fp
        if expected == 1:
            if actual == 0:
                return pr_fn
            else:
                return 1 - pr_fn

    def predict_bit(self, actual_bit, pos):
        self.oracle_calls += 1
        rnd = random.random()
        pr_fp, pr_fn = self.p_positional[pos]
        if actual_bit == 0:
            if rnd < pr_fp:
                return 1 - actual_bit
            else:
                return actual_bit
        else:
            if rnd < pr_fn:
                return 1 - actual_bit
            else:
                return actual_bit


def pr_cond_yx(y, x, pr_oracle):
    # compute Pr[Y = y | X = x]
    res = 1
    for i in range(len(x)):
        res *= pr_oracle.prob_of(x[i], y[i], i)
    return res


def pr_y(y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight):
    # compute Pr[Y = y]
    res = 0
    for s in secret_range_func(sum_weight):
        xprime = coding[s]
        pr_xprime_y = distrib_secret[s] * pr_cond_yx(y, xprime, pr_oracle)
        res += pr_xprime_y
    return res


def pr_cond_xy(
    s,
    y,
    pr_oracle,
    secret_range_func,
    coding,
    distrib_secret,
    sum_weight,
    pr_y_saved=None,
):
    # compute Pr[X = coding[s] | Y = y]
    if pr_y_saved is None:
        pr_y_saved = pr_y(
            y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
        )
    if pr_y_saved == 0:
        return 0
    return pr_cond_yx(y, coding[s], pr_oracle) * distrib_secret[s] / pr_y_saved


def pr_of_y_from_prediction(pred_y, y):
    res = 1
    for p, yval in zip(pred_y, y):
        if yval == 0:
            res *= 1 - p
        else:
            res *= p
    return res


def s_distribution_from_hard_y(
    y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
):
    distr = [0] * len(coding)
    for i, s in enumerate(secret_range_func(sum_weight)):
        pr_y_saved = pr_y(
            y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
        )
        distr[i] = pr_cond_xy(
            s,
            y,
            pr_oracle,
            secret_range_func,
            coding,
            distrib_secret,
            sum_weight,
            pr_y_saved,
        )
    return distr


def pr_cond_yx_adaptive(y, s, pr_oracle, coding_tree):
    # compute Pr[Y = y | X = e(s)]
    res = 1
    node = coding_tree
    for y_val in y:
        pos = (node.ge_flag, node.value)
        if node.ge_flag:
            expected_bit = int(s >= node.value)
        else:
            expected_bit = int(s <= node.value)
        res *= pr_oracle.prob_of(expected_bit, y_val, pos)
        if y_val == 1:
            node = node.right
        else:
            node = node.left
    return res


def pr_y_adaptive(
    y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    # compute Pr[Y = y]
    res = 0
    for s in secret_range_func(sum_weight):
        pr_xprime_y = distrib_secret[s] * pr_cond_yx_adaptive(
            y, s, pr_oracle, coding_tree
        )
        res += pr_xprime_y
    return res


def pr_cond_xy_adaptive(
    s,
    y,
    pr_oracle,
    secret_range_func,
    coding_tree,
    distrib_secret,
    sum_weight,
    pr_y_saved=None,
):
    # compute Pr[X = e(s) | Y = y]
    if pr_y_saved is None:
        pr_y_saved = pr_y_adaptive(
            y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
        )
    return (
        pr_cond_yx_adaptive(y, s, pr_oracle, coding_tree)
        * distrib_secret[s]
        / pr_y_saved
    )


def s_distribution_from_hard_y_adaptive(
    y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    # assume here that distrib_secret include probabilities for all values, i.e.
    # distrib_secret[s] is 0 for non-existent s in original distrib_secret
    distr = [0] * (2 * sum_weight + 1)
    for i, s in enumerate(secret_range_func(sum_weight)):
        distr[i] = pr_cond_xy_adaptive(
            s,
            y,
            pr_oracle,
            secret_range_func,
            coding_tree,
            distrib_secret,
            sum_weight,
            None,
        )
    return distr


def s_distribution_from_prediction_y(
    pred_y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
):
    distr = [0] * len(secret_range_func(sum_weight))
    for y in it.product(range(2), repeat=len(coding[0])):
        pr_y_saved = pr_y(
            y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
        )
        for i, s in enumerate(secret_range_func(sum_weight)):
            distr[i] += pr_cond_xy(
                s,
                y,
                pr_oracle,
                secret_range_func,
                coding,
                distrib_secret,
                sum_weight,
                pr_y_saved,
            ) * pr_of_y_from_prediction(pred_y, y)
    return distr
