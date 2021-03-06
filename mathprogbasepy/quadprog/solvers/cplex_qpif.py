from __future__ import print_function
# CPLEX interface to solve QP problems
from builtins import range
import numpy as np
import mathprogbasepy.quadprog.problem as qp
from mathprogbasepy.quadprog.solvers.solver import Solver
from mathprogbasepy.quadprog.results import QuadprogResults

import cplex as cpx


class CPLEX(Solver):
    """
    An interface for the CPLEX QP solver.
    """

    # Map of CPLEX status to CVXPY status. #TODO: add more!
    STATUS_MAP = {1: qp.OPTIMAL,
                  3: qp.PRIMAL_INFEASIBLE,
                  2: qp.DUAL_INFEASIBLE,
                  21: qp.DUAL_INFEASIBLE,
                  22: qp.PRIMAL_INFEASIBLE,
                  4: qp.PRIMAL_OR_DUAL_INFEASIBLE,
                  10: qp.MAX_ITER_REACHED,
                  101: qp.OPTIMAL,
                  103: qp.PRIMAL_INFEASIBLE,
                  107: qp.TIME_LIMIT,
                  118: qp.DUAL_INFEASIBLE}

    def solve(self, p):

        if p.P is not None:
            p.P = p.P.tocsr()

        if p.A is not None:
            # Convert Matrices in CSR format
            p.A = p.A.tocsr()

        # Get problem dimensions
        n = p.n
        m = p.m

        # Adjust infinity values in bounds
        u = np.copy(p.u)
        l = np.copy(p.l)

        # Convert infinity values to Cplex Infinity
        u = np.minimum(u, cpx.infinity)
        l = np.maximum(l, -cpx.infinity)
        if p.i_l is not None:
            i_l = np.maximum(p.i_l, -cpx.infinity)
        else:
            i_l = None
        if p.i_u is not None:
            i_u = np.minimum(p.i_u, cpx.infinity)
        else:
            i_u = None

        # Define CPLEX problem
        model = cpx.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables
        var_idx = model.variables.add(obj=p.q,
                                      lb=-cpx.infinity*np.ones(n),
                                      ub=cpx.infinity*np.ones(n))

        # Constrain integer variables if present
        # import ipdb; ipdb.set_trace()
        if p.i_idx is not None:
            for i in range(len(p.i_idx)):
                # import ipdb; ipdb.set_trace()
                model.variables.set_types(var_idx[p.i_idx[i]], 'I')
                if i_l is not None:
                    model.variables.set_lower_bounds(var_idx[p.i_idx[i]],
                                                     i_l[i])
                if i_u is not None:
                    model.variables.set_upper_bounds(var_idx[p.i_idx[i]], 
                                                     i_u[i])

        # Add constraints
        for i in range(m):  # Add inequalities
            start = p.A.indptr[i]
            end = p.A.indptr[i+1]
            row = [[p.A.indices[start:end].tolist(),
                   p.A.data[start:end].tolist()]]
            if (l[i] != -cpx.infinity) & (u[i] == cpx.infinity):
                model.linear_constraints.add(lin_expr=row,
                                             senses=["G"],
                                             rhs=[l[i]])
            elif (l[i] == -cpx.infinity) & (u[i] != cpx.infinity):
                model.linear_constraints.add(lin_expr=row,
                                             senses=["L"],
                                             rhs=[u[i]])
            elif (l[i] == -cpx.infinity) & (u[i] == cpx.infinity):
                # Dummy constraint that is always satisfied.
                model.linear_constraints.add(lin_expr=row,
                                             senses=["L"],
                                             rhs=[cpx.infinity])
            else:
                model.linear_constraints.add(lin_expr=row,
                                             senses=["R"],
                                             range_values=[l[i] - u[i]],
                                             rhs=[u[i]])

        # Set quadratic Cost
        if p.P.count_nonzero():  # Only if quadratic form is not null
            qmat = []
            for i in range(n):
                start = p.P.indptr[i]
                end = p.P.indptr[i+1]
                qmat.append([p.P.indices[start:end].tolist(),
                            p.P.data[start:end].tolist()])
            model.objective.set_quadratic(qmat)

        # Set parameters
        for param, value in self.options.items():
            if param == "verbose":
                if value == 0:
                    model.set_results_stream(None)
                    model.set_log_stream(None)
                    model.set_error_stream(None)
                    model.set_warning_stream(None)
            else:
                exec("model.parameters.%s.set(%d)" % (param, value))

        # Solve problem
        # -------------
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
        except:  # Error in the solution
            if self.options['verbose']:
                print("Error in CPLEX solution\n")
            return QuadprogResults(qp.SOLVER_ERROR, None, None, None,
                                   np.inf, None)

        # Return results
        # ---------------
        # Get status
        status = self.STATUS_MAP.get(model.solution.get_status(),
                                     qp.SOLVER_ERROR)
        # Get computation time
        cputime = end-start

        # Get total number of iterations
        total_iter = \
            int(model.solution.progress.get_num_barrier_iterations())

        if status in qp.SOLUTION_PRESENT:
            # Get objective value
            objval = model.solution.get_objective_value()

            # Get solution
            sol = np.array(model.solution.get_values())

            # Get dual values
            if p.i_idx is None:
                dual = -np.array(model.solution.get_dual_values())
            else:
                dual = None

            return QuadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:
            return QuadprogResults(status, None, None, None,
                                   cputime, total_iter)
