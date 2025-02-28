import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def lv( t_start, t_end, no, alpha=0.1, beta=0.02, gamma=0.3, delta=0.01, x0=20, y0=9,): 
    # Lotka-Volterra equations
    def lotka_volterra(t, z, alpha, beta, delta, gamma):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]
    
    t_eval = np.linspace(t_start, t_end, no)
    solution = solve_ivp(lotka_volterra, [t_start, t_end], [x0, y0], t_eval=t_eval, args=(alpha, beta, delta, gamma))
    x, y = solution.y
    lotka_volterra_array = np.vstack((x, y)).T
    return lotka_volterra_array
