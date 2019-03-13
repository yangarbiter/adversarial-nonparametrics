from abc import abstractmethod


class AttackModel():

    def __init__(self, ord):
        self.ord = ord
        super().__init__()

    @abstractmethod
    def perturb(self, X, y, eps):
        pass
