
'''
This is a very simple Pyomo Optimisation so that users can better understand how it works.
This script is also a good tool to test that pyomo installation has worked and that the solver
has been correctly added to the Python path.

Help available:
http://www.pyomo.org/installation
https://www.osti.gov/servlets/purl/1376827
https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers
'''
import pyomo.environ as pyo

# Create Model #
model = pyo.ConcreteModel()

# Create Decision Variables # 
model.x = pyo.Var( initialize=-1.2, bounds=(-2, 2) )
model.y = pyo.Var( initialize= 1.0, bounds=(-2, 2) )

# Create Parameters #
model.x_param = pyo.Param(initialize = 2.3)

# Create Objective Function #
model.obj = pyo.Objective(
expr= (1-model.x) + 100*(model.y-model.x*model.x_param), sense= pyo.minimize )

'''
 # Create a solver #
must choose from the pre-set list of compatible solvers, which can be found in pyomo documentation.
Here I have used mosek, but the user can choose one for their preferance. SCORES only needs
solvers capable of solving continuos linear programmes. A useful list of these can be found here: https://yalmip.github.io/allsolvers/
'''

opt = pyo.SolverFactory('mosek')

# Solve Model #
opt.solve(model)

# Print results #
print('x:', pyo.value(model.x))
print('y:', pyo.value(model.y))

