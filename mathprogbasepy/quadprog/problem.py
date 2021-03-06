from __future__ import absolute_import
# Interface to various QP solvers
from builtins import object
import numpy as np
import mathprogbasepy.quadprog.solvers.solvers as s

# Solver Constants
OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal inaccurate"
PRIMAL_INFEASIBLE = "primal infeasible"
PRIMAL_INFEASIBLE_INACCURATE = "primal infeasible inaccurate"
DUAL_INFEASIBLE = "dual infeasible"
DUAL_INFEASIBLE_INACCURATE = "dual infeasible inaccurate"
PRIMAL_OR_DUAL_INFEASIBLE = "primal or dual infeasible"
SOLVER_ERROR = "solver_error"
MAX_ITER_REACHED = "max_iter_reached"
TIME_LIMIT = "time_limit"

# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE]



class QuadprogProblem(object):
    """
    Defines QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	l <= A x <= u
                    i_l[i] <= x_i <= i_u[i] for i \\in i_idx
                    x_i \\in Z for i \\in i_idx

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
    i_l: numpy vector
        lower bound on integer variables
    i_u: numpy vector
        upper bound on integer variables
    i_idx: numpy vector
        index of integer variables
    """

    def __init__(self, P=None, q=None, A=None, 
                 l=None, u=None, i_idx=None, 
                 i_l=None, i_u=None,
                 x0=None):


        #
        # Get problem dimensions
        #
        if P is None:
            if q is not None:
                self.n = len(q)
            elif A is not None:
                self.n = A.shape[1]
            else:
                raise ValueError("The problem does not have any variables")
        else:
            self.n = P.shape[0]
        if A is None:
            self.m = 0
        else:
            self.m = A.shape[0]

        if i_idx is not None:
            if i_l is not None:
                if len(i_l) != len(i_idx):
                    raise ValueError("Wrong number of integer variables lower bounds")
            if i_u is not None:
                if len(i_u) != len(i_idx):
                    raise ValueError("Wrong number of integer variables upper bounds")
        self.P = P
        self.q = q
        self.A = A
        self.l = l if l is not None else -np.inf*np.ones(P.shape[0])
        self.u = u if u is not None else np.inf*np.ones(P.shape[0])
        self.i_idx = i_idx
        self.i_u = i_u
        self.i_l = i_l
        self.x0 = x0

        if x0 is not None and len(x0) != self.n:
            raise ValueError('Initial guess has wrong dimensions!')

    def solve(self, solver=s.GUROBI, **kwargs):
        """
        Solve Quadratic Program with desired solver
        """
        # Set solver
        if solver == s.GUROBI:
            from .solvers.gurobi_qpif import GUROBI
            solver = GUROBI(**kwargs)  # Initialize solver
        elif solver == s.CPLEX:
            from .solvers.cplex_qpif import CPLEX
            solver = CPLEX(**kwargs)  # Initialize solver
        elif solver == s.OSQP:
            from .solvers.osqp_qpif import OSQP
            solver = OSQP(**kwargs)  # Initialize solver
        elif solver == s.MOSEK:
            from .solvers.mosek_qpif import MOSEK
            solver = MOSEK(**kwargs)  # Initialize solver
        elif solver == s.qpOASES:
            from .solvers.qpoases_qpif import qpOASES
            solver = qpOASES(**kwargs)  # Initialize solver
        elif solver == s.OSQP_PUREPY:
            from .solvers.osqp_purepy_qpif import OSQP_PUREPY
            solver = OSQP_PUREPY(**kwargs)  # Initialize solver

        # Solve problem
        results = solver.solve(self)  # Solve problem

        return results
