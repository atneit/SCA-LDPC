from math import comb as binomial
from os import urandom

import numpy as np

from simulate import make_code, utils

# Kyber-512 params
# Q = 3329
# ETA = 3
# block_len = 256
# num_blocks = 2

# Kyber-768 params
Q = 3329
ETA = 2
block_len = 256
num_blocks = 3

# Kyber-1024 params
# Q = 3329
# ETA = 2
# block_len = 256
# num_blocks = 4


def secret_range(sum_weight):
    return range(-sum_weight * ETA, sum_weight * ETA + 1)


def secret_range_len(sum_weight):
    return 2 * sum_weight * ETA + 1


def sample_secret_coefs(n):
    ret = [0] * n
    # not the most efficient way, some bits are unused
    rnd = urandom(n)
    for i in range(n):
        r = rnd[i]
        for _ in range(ETA):
            ret[i] += r & 1
            r >>= 1
        for _ in range(ETA):
            ret[i] -= r & 1
            r >>= 1
    return ret


def coding_from_patterns(pattern, sum_weight=1):
    B = sum_weight * ETA
    if len(pattern) != (2 * B + 1):
        raise ValueError("len of pattern doesn't match sum weight")
    if type(pattern[0]) is tuple:
        return {s: p_val for s, p_val in zip(range(-B, B + 1), pattern)}
    else:
        return {s: (p_val,) for s, p_val in zip(range(-B, B + 1), pattern)}


def secret_distribution(sum_weight=1):
    B = sum_weight * ETA
    n = 2 * B
    den = 2**n
    return {s: (binomial(n, s + B) / den) for s in range(-B, B + 1)}


def gen_ldpc_matrix(sum_weight, rng_state, check_blocks):
    return make_code.make_qary_qc_parity_check_matrix(
        block_len,
        sum_weight,
        num_blocks,
        utils.make_random_state(rng_state),
        check_blocks,
    )


def to_zq_range(x):
    x = x % Q
    if x > Q / 2:
        return x - Q
    else:
        return x


def compute_ssum(s, H, check_blocks):
    l = block_len * check_blocks
    ssum = [0] * l
    for i in range(l):
        for j in range(block_len * num_blocks):
            if H[i][j] != 0:
                ssum[i] += to_zq_range(s[j // block_len][j % block_len]) * H[i][j]
    return ssum


def generate_secret():
    s = list(sample_secret_coefs(block_len) for _ in range(num_blocks))
    return s


def generate_secret_for_H(need_ssum, H, check_blocks):
    s = generate_secret()
    if need_ssum is False:
        return s, None
    ssum = compute_ssum(s, H, check_blocks)
    return s, ssum


def pattern_four_consecutive(l, idx):
    ret = [0] * l
    for i in range(l):
        ret[i] = 1 - (((i - idx) // 4) % 2)
    return tuple(ret)


single_patterns_database = {
    1: {
        2: (((0, 0), (1, 0), (0, 1), (1, 1), (0, 0)), -1),
        3: (((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)), -1),
    },
    0.995: {
        1: ((0, 1, 0, 1, 0), 0.954585307666206),
        2: (((0, 0), (1, 0), (0, 1), (1, 1), (0, 0)), 1.81774258488288),
        3: (((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)), 1.98362204455267),
    },
    0.95: {
        1: ((0, 1, 0, 1, 0), 0.713603042884044),
        2: (((0, 0), (1, 0), (0, 1), (1, 1), (0, 0)), 1.35893734442610),
        3: (((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)), 1.65239388561346),
        4: (
            ((0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)),
            1.81879316207406,
        ),
        5: (
            (
                (0, 0, 0, 0, 0),
                (1, 0, 0, 1, 1),
                (0, 1, 1, 1, 0),
                (0, 1, 0, 0, 1),
                (1, 0, 1, 0, 0),
            ),
            1.90087902706089,
        ),
    },
    0.9: {
        1: ((0, 1, 0, 1, 0), 0.531004406410719),
        2: (((0, 0), (1, 0), (0, 1), (1, 1), (0, 0)), 1.01362230968129),
        3: (((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0)), 1.32785802304240),
        # 3: (((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)), 0),
        4: (
            ((0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 1, 1), (1, 1, 0, 0), (1, 0, 0, 1)),
            1.53326842875671,
        ),
        5: (
            (
                (0, 0, 0, 0, 0),
                (1, 0, 1, 0, 1),
                (0, 1, 1, 1, 0),
                (0, 0, 0, 1, 1),
                (1, 1, 0, 0, 0),
            ),
            1.66523603579579,
        ),
    },
}

sum_patterns_database = {
    6: {
        1: (
            0.4087005109032,
            (0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        ),
        # not optimal, considered only (0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1) and (0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0)
        2: (
            0.67694657225044,
            (
                (0, 0),
                (0, 1),
                (0, 0),
                (1, 1),
                (1, 0),
                (0, 0),
                (0, 1),
                (0, 0),
                (1, 1),
                (1, 0),
                (1, 1),
                (0, 1),
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 0),
                (0, 0),
                (0, 1),
                (0, 0),
                (1, 1),
                (1, 0),
                (0, 1),
                (0, 1),
                (0, 0),
            ),
        ),
        # best random set from 1000 3-sets
        3: (
            0.869634646374502,
            (
                (0, 0, 0),
                (1, 1, 1),
                (0, 0, 0),
                (1, 0, 1),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 1),
                (0, 0, 0),
                (1, 0, 1),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 1, 1),
                (0, 0, 0),
                (1, 0, 1),
                (0, 1, 0),
                (1, 0, 1),
                (1, 1, 0),
                (0, 0, 1),
                (1, 0, 0),
                (0, 1, 1),
                (0, 0, 0),
                (1, 1, 0),
            ),
        ),
        # best random set from 170 4-sets
        4: (
            0.962966212067453,
            (
                (0, 0, 0, 0),
                (0, 1, 1, 1),
                (1, 0, 1, 1),
                (0, 1, 1, 0),
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 0, 0, 1),
                (1, 1, 0, 1),
                (0, 0, 1, 0),
                (0, 1, 1, 1),
                (1, 0, 1, 0),
                (0, 1, 1, 0),
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 0, 0, 1),
                (1, 1, 1, 1),
                (0, 0, 1, 0),
                (0, 1, 1, 1),
                (1, 0, 1, 0),
                (0, 1, 0, 1),
                (1, 1, 0, 1),
                (1, 0, 0, 0),
                (0, 1, 0, 1),
                (1, 0, 1, 0),
                (0, 1, 1, 0),
            ),
        ),
    },
}

adaptive_single_patterns_database_eta3 = {
    0.995: {
        2.5625: (
            (
                (1, 1, 1),
                (1, 1, 0),
                (1, 0),
                (0, 1),
                (0, 0, 1),
                (0, 0, 0, 1),
                (0, 0, 0, 0),
            ),
            2.33336203477099,
        ),
    }
}


def get_closest_accuracy(accuracy, accuracy_values):
    array = np.fromiter(accuracy_values, dtype=float)
    idx = (np.abs(array - accuracy)).argmin()
    return array[idx]


def get_single_patterns(eta, accuracy, num_patterns, use_closest_accuracy=False):
    assert eta == 2
    if accuracy not in single_patterns_database:
        if use_closest_accuracy:
            accuracy2 = get_closest_accuracy(accuracy, single_patterns_database.keys())
            print(f"input accuracy = {accuracy}, closest = {accuracy2}")
            accuracy = accuracy2
        else:
            raise ValueError(
                f"given accuracy ({accuracy}) is not supported, use {list(single_patterns_database.keys())}"
            )
    patterns_for_acc = single_patterns_database[accuracy]
    if num_patterns not in patterns_for_acc:
        raise ValueError(
            f"given num_patterns ({num_patterns}) is not supported, use {list(patterns_for_acc.keys())}"
        )
    return patterns_for_acc[num_patterns][0]


def get_sum_patterns(eta, num_patterns_sum, sum_weight):
    assert eta == 2
    if sum_weight not in sum_patterns_database:
        raise ValueError(
            f"given sum_weight ({sum_weight}) is not supported, use {list(sum_patterns_database.keys())}"
        )
    patterns_for_sw = sum_patterns_database[sum_weight]
    if num_patterns_sum not in patterns_for_sw:
        raise ValueError(
            f"given num_patterns_sum ({num_patterns_sum}) is not supported, use {list(patterns_for_sw.keys())}"
        )
    return patterns_for_sw[num_patterns_sum][1]


def get_restricted_single_patterns(
    eta, accuracy, num_patterns, use_closest_accuracy=False
):
    assert eta == 3
    if accuracy not in adaptive_single_patterns_database_eta3:
        if use_closest_accuracy:
            accuracy2 = get_closest_accuracy(
                accuracy, adaptive_single_patterns_database_eta3.keys()
            )
            print(f"input accuracy = {accuracy}, closest = {accuracy2}")
            accuracy = accuracy2
        else:
            raise ValueError(
                f"given accuracy ({accuracy}) is not supported, use {list(adaptive_single_patterns_database_eta3.keys())}"
            )
    patterns_for_acc = adaptive_single_patterns_database_eta3[accuracy]
    if num_patterns not in patterns_for_acc:
        raise ValueError(
            f"given num_patterns ({num_patterns}) is not supported, use {list(patterns_for_acc.keys())}"
        )
    pattern_info = patterns_for_acc[num_patterns]
    return pattern_info[0]


def get_patterns(
    eta,
    accuracy,
    num_patterns,
    num_patterns_sum,
    sum_weight,
    use_closest_accuracy=False,
):
    pattern = get_single_patterns(eta, accuracy, num_patterns, use_closest_accuracy)
    pattern_sum = get_sum_patterns(eta, num_patterns_sum, sum_weight)
    return {"pattern": pattern, "pattern_sum": pattern_sum}


def get_channel_probabilities(s_distr, ssum_distr, sum_weight, check_blocks):
    assert len(s_distr) == num_blocks
    assert len(s_distr[0]) == block_len
    ssum_len = block_len * check_blocks
    assert len(ssum_distr) == ssum_len
    B = sum_weight * ETA
    channel_output = np.zeros((block_len * num_blocks, 2 * ETA + 1)).astype(np.float32)
    channel_output_sum = np.zeros((ssum_len, 2 * B + 1)).astype(np.float32)
    for j in range(num_blocks):
        for i in range(block_len):
            channel_output[i + j * block_len] = s_distr[j][i]
    for i in range(ssum_len):
        # reverse distribution to make sure that the sum in row H is equal to 0
        channel_output_sum[i] = ssum_distr[i][::-1]
    return channel_output, channel_output_sum


def get_decoder(sum_weight, H, check_blocks, iterations):
    raise NotImplementedError("Create your own decoder based on the requirements")
    # In the following there is an example of decoders used during testing
    # For the actual attack used in the paper we used DecoderN1280R512SW6

    # if sum_weight == 3:
    #     if check_blocks != 1:
    #         raise ValueError(
    #             f"supported value for check_blocks is {{1}}, where sum_weight = {sum_weight}"
    #         )
    #     from simulate_rs import DecoderN1024R256SW3

    #     decoder = DecoderN1024R256SW3(H.astype("int8"), iterations)
    # elif sum_weight == 6:
    #     if num_blocks == 3:
    #         # Using Kyber-768
    #         if check_blocks == 1:
    #             from simulate_rs import DecoderN1024R256SW6

    #             decoder = DecoderN1024R256SW6(H.astype("int8"), iterations)
    #         elif check_blocks == 2:
    #             from simulate_rs import DecoderN1280R512SW6

    #             decoder = DecoderN1280R512SW6(H.astype("int8"), iterations)
    #         elif check_blocks == 3:
    #             from simulate_rs import DecoderN1536R768SW6

    #             decoder = DecoderN1536R768SW6(H.astype("int8"), iterations)
    #         elif check_blocks == 4:
    #             from simulate_rs import DecoderN1792R1024SW6

    #             decoder = DecoderN1792R1024SW6(H.astype("int8"), iterations)
    #         else:
    #             raise ValueError(
    #                 f"supported value for check_blocks is {{1, 2, 3, 4}}, where sum_weight = {sum_weight}"
    #             )
    #     else:
    #         raise NotImplementedError("")
    # elif sum_weight == 9:
    #     if check_blocks != 1:
    #         raise ValueError(
    #             f"supported value for check_blocks is {{1}}, where sum_weight = {sum_weight}"
    #         )
    #     from simulate_rs import DecoderN1024R256SW9

    #     decoder = DecoderN1024R256SW9(H.astype("int8"), iterations)
    # elif sum_weight == 12:
    #     if check_blocks != 1:
    #         raise ValueError(
    #             f"supported value for check_blocks is {{1}}, where sum_weight = {sum_weight}"
    #         )
    #     from simulate_rs import DecoderN1024R256V13C4B24

    #     decoder = DecoderN1024R256V13C4B24(H.astype("int8"), iterations)
    # else:
    #     raise ValueError("supported values for sum_weight is {{3, 6, 9, 12}}")
    # return decoder
