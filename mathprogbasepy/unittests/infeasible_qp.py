import scipy.sparse as spa
import numpy as np
import mathprogbasepy as mpbpy
import mathprogbasepy.quadprog.problem as problem

# Unit Test
import unittest
import numpy.testing as nptest


class infeasible_qp(unittest.TestCase):

    def setUp(self):
        # Reset random seed for repeatability
        np.random.seed(1)

        # Random Example
        n = 30
        m = 50

        # Generate random Matrices
        Pt = spa.random(n, n, density=0.6)
        P = Pt.dot(Pt.T).tocsc()
        q = np.random.randn(n)
        A = spa.random(m, n, density=0.6).tocsc()
        u = 3 + np.random.randn(m)
        l = -3 + np.random.randn(m)

        # Make random problem primal infeasible
        A[int(n/2), :] = A[int(n/2)+1, :]
        l[int(n/2)] = u[int(n/2)+1] + 10 * np.random.rand()
        u[int(n/2)] = l[int(n/2)] + 0.5

        self.p = mpbpy.QuadprogProblem(P, q, A, l, u)

    def test_infeasible_qp(self):
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
        possible_statuses = [problem.PRIMAL_INFEASIBLE,
                             problem.PRIMAL_OR_DUAL_INFEASIBLE]
        self.assertTrue(res_gurobi.status in possible_statuses)
        self.assertTrue(res_cplex.status in possible_statuses)
        self.assertTrue(res_mosek.status in possible_statuses)
        self.assertTrue(res_osqp.status in possible_statuses)
        self.assertTrue(res_qpoases.status in possible_statuses)
