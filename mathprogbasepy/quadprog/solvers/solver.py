import abc


class Solver(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.options = kwargs

    @abc.abstractmethod
    def solve(self, p):
        """Solve QP problem
        """
        pass
