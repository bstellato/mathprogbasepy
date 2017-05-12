# Interface to pure python implementation of osqp_solver
import osqppurepy as osqp
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver


class OSQP_PUREPY(Solver):
    """
    An interface for the OSQP QP solver.
    """

    def solve(self, p):

        m = osqp.OSQP()

        STATUS_MAP = {m.constant('OSQP_SOLVED'): qp.OPTIMAL,
                      m.constant('OSQP_MAX_ITER_REACHED'): qp.MAX_ITER_REACHED,
                      m.constant('OSQP_PRIMAL_INFEASIBLE'): qp.PRIMAL_INFEASIBLE,
                      m.constant('OSQP_DUAL_INFEASIBLE'): qp.DUAL_INFEASIBLE}

        if p.P is not None:
            p.P = p.P.tocsc()

        if p.A is not None:
            p.A = p.A.tocsc()

        if p.i_idx is not None:
            raise ValueError('Cannot solve MIQPs with OSQP')

        m.setup(p.P, p.q, p.A, p.l, p.u, **self.options)
        res = m.solve()

        status = STATUS_MAP.get(res.info.status_val, qp.SOLVER_ERROR)

        return QuadprogResults(status, res.info.obj_val,
                               res.x, res.y,
                               res.info.run_time, res.info.iter)
