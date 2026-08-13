#Evaluates system energy conservation by comparing net flux in due to pump vs. total energy gained by grid
def energy_change(a=a, b=b, f=f, tau=tau, zeta=zeta, I0=1e12):
    u_opt     = (n/c_light) * I0
    u_ac      = c_sound**2 * rho_0 * rho_bar**2 / 2
    flux_pref = 2 * I0
    # on-grid energy: seed + pump + acoustic, integrated over the grid
    E_grid = (u_opt*np.trapezoid(np.abs(b)**2 + np.abs(a)**2, zeta, axis=1)
              + u_ac*np.trapezoid(np.abs(f)**2, zeta, axis=1))
    grid_gain = E_grid - E_grid[0]
    # cumulative net boundary flux: pump in at left minus pump out at right
    P_net = flux_pref * (np.abs(a[:, 0])**2 - np.abs(a[:, -1])**2)     # power/area vs tau
    net_flux_in = np.concatenate(([0.0], np.cumsum(0.5*(P_net[1:] + P_net[:-1])*np.diff(tau))))
    return net_flux_in, grid_gain
  
  #example usage
flux, gain = energy_change()
plt.plot(tau, flux, label ="Net flux into grid")
plt.plot(tau, gain, label="Net energy gained in grid")

plt.xlabel("Time (s)")
plt.ylabel("Energy Flux (J)")
  
