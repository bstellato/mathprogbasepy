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

        # Convert Matrices in CSC format
        p.A = p.A.tocsc()
        p.P = p.P.tocsc()

        #  module = osqp.OSQP()
        #  module.setup(p.P, p.q, p.A, p.l, p.u, **self.options)
        res = miosqp_solve(p.P, p.Q, p.A, p.l, p.u, p.i_idx) 
        
        # TODO: Complete interface after function has been defined

        return res

        #  return QuadprogResults(res.info.status, res.info.obj_val,
                               #  res.x, res.y,
                               #  res.info.run_time, res.info.iter)
