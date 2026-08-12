#Implicit efficiency estimates efficiency with reasonable accuracy at late times

def get_eff(a_in_col = a_in_col, a_out_col = a_out_col, rad=rad, tau=tau):
    P_in  = np.trapezoid(a_in_col**2  * rad[None, :], rad, axis=1)   # pump power entering
    P_out = np.trapezoid(a_out_col**2 * rad[None, :], rad, axis=1)   # pump power leaving
    cumtrap = lambda y: np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(tau))))
    supplied = cumtrap(P_in)
    absorbed = cumtrap(P_in - P_out)
    eta = np.divide(absorbed, supplied, out=np.full_like(tau, np.nan, dtype=float), where=(supplied > 0))
    eta[0] = np.nan
    return eta
#Example usage

eff = get_eff()
plt.plot(tau, eff)
