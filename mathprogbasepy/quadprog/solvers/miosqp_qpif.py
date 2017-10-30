"""
miOSQP interface to solve MIQP problems
"""
import miosqp
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver


class MIOSQP(Solver):
    """
    An interface for the MIOSQP solver.
    """

    def solve(self, p):

        model = miosqp.MIOSQP()
        model.setup(p.P, p.q, p.A, p.l, p.u, p.i_idx, p.i_l, p.i_u, 
                    self.options)
        res_miosqp = model.solve()

        return res_miosqp
