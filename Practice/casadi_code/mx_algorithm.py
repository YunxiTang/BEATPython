from casadi import *
import numpy as np

#create a variable
x = MX.sym("x")
y = SX.sym('y', 5)

f = x**2 + 10
print(f)
