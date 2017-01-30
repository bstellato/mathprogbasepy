# mathprogbasepy
Low level interface for optimization solvers. This package is still under heavy development and is meant to be integrated to CVXPY in the future.

## Installation
`python setup.py install`


## Quadratic Programs
It is possible to define quadratic programs of the form
```
minimize     (1/2) x' P x + q' x
subject to   l <= A x <= u
```

with

```python
from mathprogbasepy import *

# Define problem data
# ...

p = QuadprogProblem(P, q, A, l, u)
results = p.solve(solver = OSQP)

```

The supported solvers at the moment are: `GUROBI`, `CPLEX`, `OSQP`.
Note that matrices `P` and `A` are in scipy sparse format. vectors `q`, `l` and `u` are numpy arrays.
