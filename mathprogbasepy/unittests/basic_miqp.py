import scipy.sparse as spa
import numpy as np
import mathprogbasepy as mpbpy


# Unit Test
import unittest
import numpy.testing as nptest


class basic_miqp(unittest.TestCase):

    def setUp(self):
        # Reset random seed for repeatability
        np.random.seed(1)

        # Random Example
        n = 30
        m = 50
        p = 5  # Number of integer variables

        # Generate random Matrices
        Pt = spa.random(n, n, density=0.6)
        P = Pt.dot(Pt.T).tocsc()
        q = np.random.randn(n)
        A = spa.random(m, n, density=0.6).tocsc()
        u = 3 + np.random.randn(m)
        l = -3 + np.random.randn(m)

        # Generate random vector of indeces
        i_idx = np.random.choice(np.arange(0, n), p, replace=False)

        self.p = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)

    def test_basic_MIQP(self):
        # Solve with GUROBI
        res_gurobi = self.p.solve(solver=mpbpy.GUROBI)

        # Solve with CPLEX
        res_cplex = self.p.solve(solver=mpbpy.CPLEX)

        # Solve with MOSEK
        res_mosek = self.p.solve(solver=mpbpy.MOSEK)

        # Assert solutions matching (GUROBI - CPLEX)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_cplex.obj_val,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.x,
                               res_cplex.x,
                               rtol=1e-4, atol=1e-4)


        # Assert solutions matching (GUROBI - MOSEK)
        nptest.assert_allclose(res_gurobi.obj_val,
                               res_mosek.obj_val,
                               rtol=1e-4, atol=1e-4)
        nptest.assert_allclose(res_gurobi.x,
                               res_mosek.x,
                               rtol=1e-4, atol=1e-4)
