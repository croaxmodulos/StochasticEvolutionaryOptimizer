import random


def select_n_random_unique_parents_from_k(n, k):
    if n > k:
        raise ValueError("n cannot be smaller than k")
    return random.sample(range(0, k), n)

