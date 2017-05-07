# mathprogbasepy
Low level interface for Mixed-Integer Quadratic Programs and Mixed-Integer Linerar Programs optimization solvers.

## Installation
```python
python setup.py install
```


## (Mixed-Integer) Quadratic Programs
It is possible to define quadratic programs of the form
```
minimize     (1/2) x' P x + q' x
subject to   l <= A x <= u
             x_i \in Z for i in I_idx
```

with

```python
from mathprogbasepy import *

# Define problem data
# ...

p = QuadprogProblem(P, q, A, l, u, i_idx)
results = p.solve(solver = OSQP)
```

The current version is `0.1.1`

The supported solvers at the moment are: `OSQP`, `GUROBI`, `CPLEX`, `MOSEK`.


Matrices `P` and `A` are in scipy sparse format. vectors `q`, `l` and `u` are numpy arrays.
