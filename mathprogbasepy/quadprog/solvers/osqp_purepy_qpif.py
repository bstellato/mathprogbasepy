# Interface to pure python implementation of osqp_solver
import osqppurepy as osqp
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver


class OSQP_PUREPY(Solver):
    """
    An interface for the OSQP QP solver.
    """

    def solve(self, p):

        if p.i_idx is not None:
            raise ValueError('Cannot solve MIQPs with OSQP')

        if p.P is not None:
            p.P = p.P.tocsc()

        if p.A is not None:
            p.A = p.A.tocsc()

        m = osqp.OSQP()
        m.setup(p.P, p.q, p.A, p.l, p.u, **self.options)
        res = m.solve()

        return QuadprogResults(res.info.status, res.info.obj_val,
                               res.x, res.y,
                               res.info.run_time, res.info.iter)
