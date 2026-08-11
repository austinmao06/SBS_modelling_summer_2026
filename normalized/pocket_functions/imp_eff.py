"""Implicit Efficiency:
rough estimate of efficiency that is reliable at higher times (takes Esupplied - Eout as Egained)
Necessary because the equations with the dropped terms contain no internal energy conservation
When seed >> pump, this is a good estimate
"""
def get_eff(a = a, tau=tau):
    a_in  = a[:, 0]**2            # pump entering the domain
    a_out = a[:, -1]**2           # pump leaving the domain (depleted)
    cumtrap = lambda y: np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(tau))))
    supplied = cumtrap(a_in)
    absorbed = cumtrap(a_in - a_out)
    eta = np.divide(absorbed, supplied, out=np.full_like(tau, np.nan, dtype=float), where=(supplied > 0))
    eta[0] = np.nan
    return eta
