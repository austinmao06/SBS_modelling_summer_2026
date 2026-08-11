"""Converts true time to tau time"""

def convert_to_tau_time(true_time):
    return w * true_time

#Usage


ti = np.argmin(np.abs(tau - convert_to_tau_time(127e-9))) #finds nearest tau index to 127ns true time

plt.plot(zeta, b[ti,]**2)

"""
Note that the prominent issue with normalized as a whole is that one must calculate fitting values for kNorm, wNorm, etc prior to running the simulation
to ensure that the necessary z and t dimensions are even met
"""
