# Interface to various QP solvers
import numpy as np
import solvers.solvers as s

# Solver Constants
OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
SOLVER_ERROR = "solver_error"
MAX_ITER_REACHED = "max_iter_reached"

# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL]
# Statuses that indicate the problem is infeasible or unbounded.
INF_OR_UNB = [INFEASIBLE, UNBOUNDED]


class QuadprogProblem(object):
    """
    Defines QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	l <= A x <= u
                    x_i \in Z for i \in i_idx

    Attributes
    ----------
    P: scipy sparse matrix
        quadratic cost matrix
    q: numpy vector
        linear cost vector
    A: scipy sparse matrix
        constraints matrix
    l: numpy vector
        constraints lower bound
    u: numpy vector
        constraints upper bound
    i_idx: numpy vector
        index of integer variables
    """

    def __init__(self, P, q, A, l=None, u=None, i_idx=None):
        self.n = P.shape[0]
        self.m = A.shape[0]
        self.P = P
        self.q = q
        self.A = A
        self.l = l if l is not None else -np.inf*np.ones(P.shape[0])
        self.u = u if u is not None else np.inf*np.ones(P.shape[0])
        self.i_idx = i_idx

    def solve(self, solver=s.GUROBI, **kwargs):
        """
        Solve Quadratic Program with desired solver
        """

        # Set solver
        if solver == s.GUROBI:
            from solvers.gurobi_qpif import GUROBI
            solver = GUROBI(**kwargs)  # Initialize solver
        elif solver == s.CPLEX:
                from solvers.cplex_qpif import CPLEX
                solver = CPLEX(**kwargs)  # Initialize solver
        elif solver == s.OSQP:
                from solvers.osqp_qpif import OSQP
                solver = OSQP(**kwargs)  # Initialize solver
        elif solver == s.OSQP_PUREPY:
                from solvers.osqp_purepy_qpif import OSQP_PUREPY
                solver = OSQP_PUREPY(**kwargs)  # Initialize solver

        # Solve problem
        results = solver.solve(self)  # Solve problem

        return results
