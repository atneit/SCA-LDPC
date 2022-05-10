from math import isinf
from collections import defaultdict


def increment_d_values(d_values, start_idx):
    for i in range(start_idx, -1, -1):
        d_values[i] += 1
        if d_values[i] <= B:
            return 0
        elif i == 0:
            return 1
        else:
            d_values[i] = -B


def send_d_values(connected_vars, d_values, beta, check_idx):
    print(d_values)
    s = sum(v[d + B] for v, d in zip(connected_vars, d_values))
    print(s)
    for i in range(len(connected_vars)):
        d = d_values[i]
        conf_sum = s - connected_vars[i][d + B]
        # \beta_{ij}, the first index is index of check variable, the second one is the position
        # of variable inside array of connected variables of check node. TODO: this is probably should
        # be changed.
        b_ijd = beta[(check_idx, i)][d + B]
        beta[(check_idx, i)][d + B] = min(b_ijd, conf_sum)


# all variables have values in [-B, ..., B]
B = 1

# variable is array of length 2*B+1. v[i + B] returns LLR of Pr[v = i]
# connected_vars = check.variables(check_idx)
check_idx = 0
# connected_vars = [[float('inf'), 0.1, 0.5, 0.3, 0.2], [0.4, 0.4, 0.4, 0.4, 0.4], [0.4, 0.6, 0.3, 0.2, float('inf')]]
connected_vars = [[float("inf"), 0.1, 0.5], [0.4, 0.4, 0.4], [0.3, 0.2, float("inf")]]
l = len(connected_vars)
beta = defaultdict(lambda: [float("inf")] * (2 * B + 1))

d_values = [-B] * l
finish = False
while not finish:
    d = 0
    for i in range(0, l - 1):
        v = connected_vars[i]
        # check for infinity
        while isinf(v[d_values[i] + B]) and not finish:
            finish = finish or increment_d_values(d_values, i)
        d += d_values[i]
    v = connected_vars[l - 1]
    if d >= -B and d <= B and not isinf(v[-d + B]):
        # sum(d_values) is equal to 0
        d_values[l - 1] = -d
        send_d_values(connected_vars, d_values, beta, check_idx)
    finish = increment_d_values(d_values, l - 2)
print(beta)
