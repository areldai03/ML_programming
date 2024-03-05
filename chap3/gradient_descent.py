import numpy as np

a = 0.2
x = 1.0

def grad(x):
    return 3*x**2

for i in range(50):
    print(f'反復回数:{i}, x={x}')
    g = grad(x)
    x = x - a * g