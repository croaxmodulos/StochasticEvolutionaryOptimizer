import abc


class RandNumber(abc.ABC):
    @abc.abstractmethod
    def NextRand(self):
        pass


class RandInt(RandNumber):
    def __int__(self, seed=5):
        self.random = seed

    def NextRand(self):
        pass
