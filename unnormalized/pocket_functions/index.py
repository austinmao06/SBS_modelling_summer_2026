#Kinda trivial but returns index of some time

index(tau = tau, t = t):
    return np.argmin(np.abs(tau - t))

#Example Usage
t = 111.1e-9
ti = index(t=t)
