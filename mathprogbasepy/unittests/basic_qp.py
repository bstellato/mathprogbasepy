import scipy.sparse as spa
import numpy as np
import mathprogbasepy as mpbpy


# Unit Test
import unittest
import numpy.testing as nptest


class basic_qp(unittest.TestCase):

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

        self.p = mpbpy.QuadprogProblem(P, q, A, l, u)

    def test_basic_QP(self):
        # Solve with GUROBI
        res_gurobi = self.p.solve(solver=mpbpy.GUROBI)

        # Solve with CPLEX
        res_cplex = self.p.solve(solver=mpbpy.CPLEX)

        # Solve with MOSEK
        res_mosek = self.p.solve(solver=mpbpy.MOSEK)

        # Solve with OSQP
        res_osqp = self.p.solve(solver=mpbpy.OSQP, polish=True)

        # Solve with qpOASES
        res_qpoases = self.p.solve(solver=mpbpy.qpOASES)


        # Assert solutions matching (GUROBI - CPLEX)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_cplex.obj_val,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.x,
                               res_cplex.x,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.y,
                               res_cplex.y,
                               rtol=1e-4, atol=1e-4)

        # Assert solutions matching (GUROBI - MOSEK)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_mosek.obj_val,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.x,
                               res_mosek.x,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.y,
                               res_mosek.y,
                               rtol=1e-4, atol=1e-4)

        # Assert solutions matching (GUROBI - OSQP)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_osqp.obj_val,
                               rtol=1e-3, atol=1e-3)
        nptest.assert_allclose(res_gurobi.x,
                               res_osqp.x,
                               rtol=1e-3, atol=1e-3)
        nptest.assert_allclose(res_gurobi.y,
                               res_osqp.y,
                               rtol=1e-3, atol=1e-3)

        # Assert solutions matching (GUROBI - qpOASES)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_qpoases.obj_val,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.x,
                               res_qpoases.x,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.y,
                               res_qpoases.y,
                               rtol=1e-4, atol=1e-4)
