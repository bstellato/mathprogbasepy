from builtins import object
import abc
from future.utils import with_metaclass


class Solver(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, **kwargs):
        self.options = kwargs

        if 'verbose' not in self.options:
            self.options['verbose'] = True

    @abc.abstractmethod
    def solve(self, p):
        """Solve QP problem
        """
        pass
