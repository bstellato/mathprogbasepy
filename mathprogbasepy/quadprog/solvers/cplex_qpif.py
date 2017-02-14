from __future__ import print_function
# CPLEX interface to solve QP problems
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
                  3: qp.INFEASIBLE,
                  2: qp.UNBOUNDED,
                  10: qp.MAX_ITER_REACHED,
                  101: qp.OPTIMAL,
                  103: qp.INFEASIBLE,
                  118: qp.UNBOUNDED}

    def solve(self, p):

        # Convert Matrices in CSR format
        p.A = p.A.tocsr()
        p.P = p.P.tocsr()

        # Get problem dimensions
        n = p.P.shape[0]
        m = p.A.shape[0]

        # Adjust infinity values in bounds
        u = np.copy(p.u)
        l = np.copy(p.l)

        # Convert infinity values to Cplex Infinity
        u = np.minimum(u, cpx.infinity)
        l = np.maximum(l, -cpx.infinity)

        # Define CPLEX problem
        model = cpx.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables
        model.variables.add(obj=p.q,
                            lb=-cpx.infinity*np.ones(n),
                            ub=cpx.infinity*np.ones(n))  # Linear obj part

        # Constrain integer variables if present
        if p.i_idx is not None:
            for i in p.i_idx:
                model.variables.set_types(i, 'I')

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
        for param, value in self.options.iteritems():
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

        if (status != qp.SOLVER_ERROR) & (status != qp.INFEASIBLE):
            # Get objective value
            objval = model.solution.get_objective_value()

            # Get solution
            sol = np.array(model.solution.get_values())

            # Get dual values
            if p.i_idx is None:
                dual = -np.array(model.solution.get_dual_values())
            else:
                dual = None

            # Get computation time
            cputime = end-start

            # Get total number of iterations
            total_iter = \
                int(model.solution.progress.get_num_barrier_iterations())

            return QuadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:
            return QuadprogResults(status, None, None, None,
                                   cputime, None)
