# Quadprog results class
from builtins import object


class QuadprogResults(object):

    """
    Stores results of a QP solver
    """

    def __init__(self, status, obj_val, x, y, cputime, total_iter):
        self.status = status
        self.obj_val = obj_val
        self.x = x
        self.y = y
        self.cputime = cputime
        self.total_iter = total_iter
