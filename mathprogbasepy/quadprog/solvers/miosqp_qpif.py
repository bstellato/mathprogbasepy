"""
miOSQP interface to solve MIQP problems
"""
import osqp
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver


class MIOSQP(Solver):
    """
    An interface for the MIOSQP solver.
    """

    def solve(self, p):

        res = miosqp_solve(p.P, p.Q, p.A, p.l, p.u, p.i_idx)

        return res
