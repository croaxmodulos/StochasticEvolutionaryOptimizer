import random


class RandomUniqueParentsSelection:
    def __init__(self, num_new_parents):
        self.num_new_parents = num_new_parents

    def select_random_unique_parents(self, num_total_parents):
        if self.num_new_parents > num_total_parents:
            raise ValueError("number of total parents cannot be smaller than the number of new parents")

        return random.sample(range(0, num_total_parents), self.num_new_parents)

