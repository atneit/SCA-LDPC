import numpy as np


def calc_ds(input: np.array):
    """
    Calculates the distance spectrum of the input array

    >>> a = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    >>> calc_ds(a)
    array([0, 1, 1, 2, 1, 1])
    """
    li = len(input)
    l = ((li) // 2) + 1
    output = np.zeros(l, dtype=int)
    ones = [i for (i, x) in enumerate(input) if x]
    for (i, first) in enumerate(ones):
        for second in ones[(i + 1) :]:
            d = second - first
            d = min(d, li - d)
            output[d] += 1
    return output


def check_ds_addition_limit(input: np.array, ds: np.array, add: int, limit: int):
    """
    Calculates the effect on the distance spectrum if a bit is flipped at position 'add'

    >>> a = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    >>> ds = calc_ds(a)
    >>> check_ds_addition_limit(a, ds, 6, 2)
    array([0, 1, 1, 2, 1, 1])
    """
    li = len(input)
    ds = ds.copy()
    ones = [i for (i, x) in enumerate(input) if x]
    for (i, first) in enumerate(ones):
        d = abs(add - first)
        d = min(d, li - d)
        ds[d] += 1
        if ds[d] > limit:
            return False

    return ds


def gen_array_ds_multiplicity(
    length: int, weight: int, max_multiplicity: int, rng: np.random.RandomState
):
    """
    Creates a random array with 'weight' number of set positions. Additionally
    guarantees the distance spectrum multiplicity to be maximum 'max_multiplicity'.

    >>> from . import utils
    >>> rng = utils.make_random_state(0)
    >>> a = gen_array_ds_multiplicity(10, 3, 1, rng)
    >>> ds = calc_ds(a)
    >>> (a, ds)
    (array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0]), array([0, 0, 1, 1, 0, 1]))
    """
    output = np.zeros(length, dtype=int)
    first = rng.random_integers(0, length - 1)
    output[first] = 1
    ds = calc_ds(output)
    w = 1
    while w < weight:
        # Get random addition
        next = rng.random_integers(0, length - 1)
        # Calculate effect on distance spectrum
        new_ds = check_ds_addition_limit(output, ds, next, max_multiplicity)
        # Check if valid
        if isinstance(new_ds, np.ndarray):
            ds = new_ds
            output[next] = 1
            w += 1
    return output


if __name__ == "__main__":
    print("In module products __package__, __name__ ==", __package__, __name__)
    import utils

    rng = utils.make_random_state(0)
    a = gen_array_ds_multiplicity(10, 3, 1, rng)
    ds = calc_ds(a)
