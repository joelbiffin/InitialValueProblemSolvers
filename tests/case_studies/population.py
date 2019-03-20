f = lambda u, t: np.array([
    2*u[0] - u[0]*u[1],
    -u[1] + u[0]*u[1]
])

true_value = lambda t: np.array([
    exp(2*t),
    exp(-t)
])