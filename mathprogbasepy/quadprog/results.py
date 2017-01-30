# Quadprog results class
class QuadprogResults(object):

    """
    Stores results of a QP solver
    """

    def __init__(self, status, objval, x, y, cputime, total_iter):
        self.status = status
        self.objval = objval
        self.x = x
        self.y = y
        self.cputime = cputime
        self.total_iter = total_iter
