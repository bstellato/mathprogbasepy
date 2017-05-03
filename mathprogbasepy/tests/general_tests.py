#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
from __future__ import print_function
import sys
import scipy.io as spio
import scipy.sparse as spspa
import scipy as sp
import numpy as np
import ipdb
import mathprogbasepy as mpbpy

np.random.seed(3)

# Random Example
n = 3
m = 5
# Generate random Matrices
Pt = sp.randn(n, n)
P = spspa.csc_matrix(np.dot(Pt.T, Pt))
q = sp.randn(n)
A = spspa.csc_matrix(sp.randn(m, n))
u = 3 + sp.randn(m)
l = -3 + sp.randn(m)

p = mpbpy.QuadprogProblem(P, q, A, l, u)

print("\nSolve with CPLEX")
print("-----------------")
resultsCPLEX = p.solve(solver=mpbpy.CPLEX, verbose=True)

print("\nSolve with GUROBI")
print("-----------------")
resultsGUROBI = p.solve(solver=mpbpy.GUROBI, verbose=True)

print("\nSolve with MOSEK")
print("-----------------")
resultsMOSEK = p.solve(solver=mpbpy.MOSEK, verbose=True)

# Solve with OSQP.
print("\nSolve with OSQP")
print("-----------------")
resultsOSQP = p.solve(solver=mpbpy.OSQP)

if resultsGUROBI.status != 'solver_error':
    print("\n")
    print("Comparison CPLEX - GUROBI")
    print("-------------------------")
    print("Difference in objective value %.5f" % \
        np.linalg.norm(resultsCPLEX.obj_val - resultsGUROBI.obj_val))
    print("Norm of solution difference %.5f" % \
        np.linalg.norm(resultsCPLEX.x - resultsGUROBI.x))
    print("Norm of dual difference %.5f" % \
        np.linalg.norm(resultsCPLEX.y - resultsGUROBI.y))

    print("\n")
    print("Comparison MOSEK - GUROBI")
    print("-------------------------")
    print("Difference in objective value %.5f" % \
        np.linalg.norm(resultsMOSEK.obj_val - resultsGUROBI.obj_val))
    print("Norm of solution difference %.5f" % \
        np.linalg.norm(resultsMOSEK.x - resultsGUROBI.x))
    print("Norm of dual difference %.5f" % \
        np.linalg.norm(resultsMOSEK.y - resultsGUROBI.y))

    print("\n")
    print("Comparison OSQP - GUROBI")
    print("-------------------------")
    print("Difference in objective value %.5f" % \
        np.linalg.norm(resultsOSQP.obj_val - resultsGUROBI.obj_val))
    print("Norm of solution difference %.5f" % \
        np.linalg.norm(resultsOSQP.x - resultsGUROBI.x))
    print("Norm of dual difference %.5f" % \
        np.linalg.norm(resultsOSQP.y - resultsGUROBI.y))

else:
    print("Problem is primal infeasible or dual infeasible")



# # Solve with mosek and CVXPY
# import cvxpy
# x = cvxpy.Variable(n)
# objective = cvxpy.Minimize(.5 * cvxpy.quad_form(x, P) + q * x)
# constraints = [A * x <= u, l <= A * x]
# problem = cvxpy.Problem(objective, constraints)
# problem.solve(solver=cvxpy.MOSEK, verbose=True)
# dual_cvxpy = (constraints[0].dual_value - constraints[1].dual_value).A1
# primal_cvxpy = x.value.A1
#
# print("\n")
# print("Comparison CVXPY - GUROBI")
# print("-------------------------")
# print("Difference in objective value %.5f" % \
#     np.linalg.norm(resultsGUROBI.obj_val - objective.value))
# print("Norm of solution difference %.5f" % \
#     np.linalg.norm(resultsGUROBI.x - primal_cvxpy))
# print("Norm of dual difference %.5f" % \
#     np.linalg.norm(resultsGUROBI.y - dual_cvxpy))
#
# print("\n")
# print("Comparison CVXPY - MOSEK")
# print("-------------------------")
# print("Difference in objective value %.5f" % \
#     np.linalg.norm(resultsMOSEK.obj_val - objective.value))
# print("Norm of solution difference %.5f" % \
#     np.linalg.norm(resultsMOSEK.x - primal_cvxpy))
# print("Norm of dual difference %.5f" % \
#     np.linalg.norm(resultsMOSEK.y - dual_cvxpy))
