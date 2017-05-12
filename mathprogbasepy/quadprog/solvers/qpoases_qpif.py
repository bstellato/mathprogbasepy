# OSQP interface to solve QP problems
import qpoases
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver
import mathprogbasepy.quadprog.problem as qp
import numpy as np


class qpOASES(Solver):
    """
    An interface for the qpOASES QP solver.
    """

    # Get return value class
    _PyReturnValue = qpoases.PyReturnValue()
    STATUS_MAP = {_PyReturnValue.SUCCESSFUL_RETURN: qp.OPTIMAL,
                  _PyReturnValue.INIT_FAILED_INFEASIBILITY: qp.PRIMAL_INFEASIBLE,
                  _PyReturnValue.INIT_FAILED_UNBOUNDEDNESS: qp.DUAL_INFEASIBLE,
                  _PyReturnValue.MAX_NWSR_REACHED: qp.MAX_ITER_REACHED,
                  _PyReturnValue.INIT_FAILED: qp.SOLVER_ERROR
                  }

    def solve(self, p):

        if p.P is not None:
            P = np.ascontiguousarray(p.P.todense())

        if p.A is not None:
            A = np.ascontiguousarray(p.A.todense())

        if p.i_idx is not None:
            raise ValueError('Cannot solve MIQPs with qpOASES')

        # Define contiguous array vectors
        q = np.ascontiguousarray(p.q)
        l = np.ascontiguousarray(p.l)
        u = np.ascontiguousarray(p.u)


        # Solve with qpOASES
        qpoases_m = qpoases.PyQProblem(p.n, p.m)
        options = qpoases.PyOptions()

        if not self.options['verbose']:
            options.printLevel = qpoases.PyPrintLevel.NONE

        for param, value in self.options.items():
            if param == 'verbose':
                if value is False:
                    options.printLevel = qpoases.PyPrintLevel.NONE
            elif param == 'cputime':
                qpoases_cpu_time = np.array([value])
            elif param == 'nWSR':
                qpoases_nWSR = np.array([value])
            else:
                exec("options.%s = %s" % (param, value))

        qpoases_m.setOptions(options)

        if 'cputime' not in self.options:
            # Set default to max 10 seconds in runtime
            qpoases_cpu_time = np.array([10.])

        if 'nWSR' not in self.options:
            # Set default to max 1000 working set recalculations
            qpoases_nWSR = np.array([1000])

        # Set number of working set recalculations
        status = qpoases_m.init(P, q, A, None, None, l, u,
                                qpoases_nWSR, qpoases_cpu_time)

        # Check status
        status = self.STATUS_MAP.get(status, qp.SOLVER_ERROR)

        # run_time
        run_time = qpoases_cpu_time[0]

        # number of iterations
        niter = qpoases_nWSR[0]

        if status in qp.SOLUTION_PRESENT:
            x = np.zeros(p.n)
            y_temp = np.zeros(p.n + p.m)
            obj_val = qpoases_m.getObjVal()
            qpoases_m.getPrimalSolution(x)
            qpoases_m.getDualSolution(y_temp)
            
            # Change sign and take only last part of y (No bounds on x in our formulation)
            y = -y_temp[p.n:]

            return QuadprogResults(status, obj_val,
                                   x, y,
                                   run_time, niter)
        else:
            return QuadprogResults(status, None, None, None,
                                   run_time, niter)
