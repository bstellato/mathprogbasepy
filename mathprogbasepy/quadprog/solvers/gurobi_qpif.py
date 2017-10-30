from __future__ import print_function
# GUROBI interface to solve QP problems
from builtins import range
import numpy as np
from mathprogbasepy.quadprog.results import QuadprogResults
from mathprogbasepy.quadprog.solvers.solver import Solver
import mathprogbasepy.quadprog.problem as qp

import gurobipy as grb


class GUROBI(Solver):
    """
    An interface for the Gurobi QP solver.
    """

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: qp.OPTIMAL,
                  3: qp.PRIMAL_INFEASIBLE,
                  5: qp.DUAL_INFEASIBLE,
                  4: qp.PRIMAL_OR_DUAL_INFEASIBLE,
                  6: qp.SOLVER_ERROR,
                  7: qp.MAX_ITER_REACHED,
                  8: qp.SOLVER_ERROR,
                  9: qp.TIME_LIMIT,
                  10: qp.SOLVER_ERROR,
                  11: qp.SOLVER_ERROR,
                  12: qp.SOLVER_ERROR,
                  13: qp.SOLVER_ERROR}

    def solve(self, p):

        if p.A is not None:
            # Convert Matrices in CSR format
            p.A = p.A.tocsr()

        if p.P is not None:
            # Convert P matrix to COO format
            p.P = p.P.tocoo()

        # Get problem dimensions
        n = p.n
        m = p.m

        # Adjust infinity values in bounds
        u = np.copy(p.u)
        l = np.copy(p.l)
        
        if p.i_l is not None:
            i_l = np.copy(p.i_l)
        else:
            i_l = None
        if p.i_u is not None:
            i_u = np.copy(p.i_u)
        else:
            i_u = None

        for i in range(m):
            if u[i] >= 1e20:
                u[i] = grb.GRB.INFINITY
            if l[i] <= -1e20:
                l[i] = -grb.GRB.INFINITY

        if i_u is not None:
            for i in range(len(i_u)):
                if i_u[i] >= 1e20:
                    i_u[i] = grb.GRB.INFTY
        if i_l is not None:
            for i in range(len(i_l)):
                if i_l[i] <= -1e20:
                    i_l[i] = -grb.GRB.INFTY
       
        # Create a new model
        model = grb.Model("qp")

        # Add variables
        for i in range(n):
            model.addVar(ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)
        model.update()
        x = model.getVars()

        # Constrain integer variables if present
        if p.i_idx is not None:
            for i in range(len(p.i_idx)):
                x[p.i_idx[i]].setAttr("vtype", 'I')
                if i_l is not None:
                    x[p.i_idx[i]].setAttr("lb", i_l[i])
                if i_u is not None:
                    x[p.i_idx[i]].setAttr("ub", i_u[i])

        model.update()

        # Set initial guess if passed
        if p.x0 is not None:
            for i in range(n):
                x[i].start = p.x0[i]

        model.update()

        # Add inequality constraints: iterate over the rows of A
        # adding each row into the model
        if p.A is not None:
            for i in range(m):
                start = p.A.indptr[i]
                end = p.A.indptr[i+1]
                variables = [x[j] for j in p.A.indices[start:end]]  # Get nnz
                coeff = p.A.data[start:end]
                expr = grb.LinExpr(coeff, variables)
                if (np.abs(l[i] - u[i]) < 1e-08):
                    model.addConstr(expr, grb.GRB.EQUAL, u[i])
                elif (l[i] == -grb.GRB.INFINITY) & (u[i] == grb.GRB.INFINITY):
                    # Dummy constraint that is always satisfied.
                    # Gurobi crashes if both constraints in addRange function
                    # are infinite.
                    model.addConstr(0.*expr, grb.GRB.LESS_EQUAL, 10.)
                else:
                    model.addRange(expr, lower=l[i], upper=u[i])

        # Define objective
        if p.P is not None:
            obj = grb.QuadExpr()  # Set quadratic part
            if p.P.count_nonzero():  # If there are any nonzero elms in P
                for i in range(p.P.nnz):
                    obj.add(.5*p.P.data[i]*x[p.P.row[i]]*x[p.P.col[i]])
            obj.add(grb.LinExpr(p.q, x))  # Add linear part
            model.setObjective(obj)  # Set objective

        # Set parameters
        # if verbose null, suppress it first
        if 'verbose' in self.options:
            if self.options['verbose'] == 0:
                model.setParam("OutputFlag", 0)
        # Set other parameters
        for param, value in self.options.items():
            if param is not "verbose":
                model.setParam(param, value)

        # Update model
        model.update()

        # Solve problem
        # -------------
        try:
            # Solve
            model.optimize()
        except:  # Error in the solution
            if self.options['verbose']:
                print("Error in GUROBI solution\n")
            return QuadprogResults(qp.SOLVER_ERROR, None, None, None,
                                   np.inf, None)

        # Return results
        # --------------

        # Get status
        status = self.STATUS_MAP.get(model.Status, qp.SOLVER_ERROR)

        if status in qp.SOLUTION_PRESENT:
            # Get objective value
            objval = model.objVal

            # Get solution
            sol = np.array([x[i].X for i in range(n)])

            if p.i_idx is None:
                # Get dual variables  (Gurobi uses swapped signs (-1))
                constrs = model.getConstrs()
                dual = -np.array([constrs[i].Pi for i in range(m)])
            else:
                dual = None

            # Get computation time
            cputime = model.Runtime

            # Total Number of iterations
            total_iter = model.BarIterCount

            return QuadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:  # Error
            # Get computation time
            cputime = model.Runtime

            return QuadprogResults(status, None, None, None,
                                   cputime, None)
