# OSQP interface to solve QP problems
import qpoases
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver


class qpOASES(Solver):
    """
    An interface for the qpOASES QP solver.
    """

    STATUS_MAP = {0: qp.OPTIMAL}

    def solve(self, p):

        if p.P is not None:
            p.P = p.P.tocsc()

        if p.A is not None:
            p.A = p.A.tocsc()

        if p.i_idx is not None:
            raise ValueError('Cannot solve MIQPs with qpOASES')

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

        # The maximum cpu time has to be initialized as an array
        # and then is returned by changing this value
        qpoases_cpu_time = np.array([10.])

        if qpoases_cpu_time is None:
            qpoases_cpu_time = np.array([10.])

        if nWSR is None:
            nWSR = np.array([1000])

        # Set number of working set recalculations
        status = qpoases_m.init(P.todense(), q, A.todense(), None, None, l, u,
                                qpoases_nWSR, qpoases_cpu_time)
        x = np.zeros(p.n)
        y = np.zeros(p.m)
        obj_val = qpoases_m.getObjVal()
        qpoases_m.getPrimalSolution(x)
        qpoases_m.getDualSolution(x)

        # run_time
        run_time = qpoases_cpu_time[0]

        # number of iterations
        niter = qpoases_nWSR[0]

        # Print results
        print('Norm of x value difference = %.4f' %
              np.linalg.norm(res_osqp_x - res_qpoases_x))
        print('Norm of objective value difference = %.4f' %
              abs(res_osqp_objval - res_qpoases_objval))
        print('Time qpoases %.4f' % qpoases_cpu_time[0])



        return QuadprogResults(status, obj_val,
                               x, y,
                               run_time, niter)
