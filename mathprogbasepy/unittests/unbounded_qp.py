import scipy.sparse as spa
import numpy as np
import mathprogbasepy as mpbpy
import mathprogbasepy.quadprog.problem as problem

# Unit Test
import unittest
import numpy.testing as nptest


class unbounded_qp(unittest.TestCase):

    def setUp(self):

        # Dual infeasible example
        P = spa.csc_matrix(np.diag(np.array([4., 0.])))
        q = np.array([0., 2])
        A = spa.csc_matrix([[1., 1.], [-1., 1.]])
        l = np.array([-np.inf, -np.inf])
        u = np.array([2., 3.])

        self.p = mpbpy.QuadprogProblem(P, q, A, l, u)

    def test_unbounded_qp(self):
        # Solve with GUROBI
        res_gurobi = self.p.solve(solver=mpbpy.GUROBI)

        # Solve with CPLEX
        res_cplex = self.p.solve(solver=mpbpy.CPLEX)

        # Solve with MOSEK
        res_mosek = self.p.solve(solver=mpbpy.MOSEK)

        # Solve with OSQP
        res_osqp = self.p.solve(solver=mpbpy.OSQP)

        # Solve with qpOASES
        res_qpoases = self.p.solve(solver=mpbpy.qpOASES)

        # Assert all statuses are infeasible
        possible_statuses = [problem.DUAL_INFEASIBLE,
                             problem.PRIMAL_OR_DUAL_INFEASIBLE]
        self.assertTrue(res_gurobi.status in possible_statuses)
        self.assertTrue(res_cplex.status in possible_statuses)
        self.assertTrue(res_mosek.status in possible_statuses)
        self.assertTrue(res_osqp.status in possible_statuses)
        self.assertTrue(res_qpoases.status in possible_statuses)
