import itertools as it


class BaseOracle:
    def prob_of(self, expected, actual, pos):
        raise NotImplementedError()


class SimpleOracle(BaseOracle):
    # p: accuracy of oracle
    def __init__(self, p):
        self.p = p

    def prob_of(self, expected, actual, pos):
        if actual == expected:
            return self.p
        else:
            return 1 - self.p


class FalsePositiveNegativePositionalOracle(BaseOracle):
    # as input p_arr should be array of tuples, where tuple i contains a pair of probability of false positive and
    # false negative, resp., for pos == i
    def __init__(self, p_arr):
        self.p_arr = p_arr

    def prob_of(self, expected, actual, pos):
        pr_fp, pr_fn = self.p_arr[pos]
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


# TODO: fix *_adaptive with probability oracles
def pr_cond_yx_adaptive(y, s, p, coding_tree):
    # compute Pr[Y = y | X = e(s)]
    res = 1
    node = coding_tree
    for y_val in y:
        if s < node.value:
            # y_val should be 1
            if y_val == 1:
                res *= p
                node = node.left
            else:
                res *= 1 - p
                node = node.right
        else:
            # y_val should be 0
            if y_val == 1:
                res *= 1 - p
                node = node.left
            else:
                res *= p
                node = node.right
    return res


def pr_y_adaptive(y, p, secret_range_func, coding_tree, distrib_secret, sum_weight):
    # compute Pr[Y = y]
    res = 0
    for s in secret_range_func(sum_weight):
        pr_xprime_y = distrib_secret[s] * pr_cond_yx_adaptive(y, s, p, coding_tree)
        res += pr_xprime_y
    return res


def pr_cond_xy_adaptive(
    s,
    y,
    p,
    secret_range_func,
    coding_tree,
    distrib_secret,
    sum_weight,
    pr_y_saved=None,
):
    # compute Pr[X = e(s) | Y = y]
    if pr_y_saved is None:
        pr_y_saved = pr_y_adaptive(
            y, p, secret_range_func, coding_tree, distrib_secret, sum_weight
        )
    return pr_cond_yx_adaptive(y, s, p, coding_tree) * distrib_secret[s] / pr_y_saved


def s_distribution_from_hard_y_adaptive(
    y, p, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    distr = [0] * len(distrib_secret)
    for i, s in enumerate(secret_range_func(sum_weight)):
        distr[i] = pr_cond_xy_adaptive(
            s,
            y,
            p,
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
