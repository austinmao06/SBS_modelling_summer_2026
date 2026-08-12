#Returns net energy flux into grid and energy gained within grid. Checks energy conservation

def energy_change(b_out=b, a_out=a, f_out=f, a_in_col=a_in_col, a_out_col=a_out_col,
                  tau=tau, times_to_save=times_to_save, zeta=zeta, rad=rad, I0=1e12):
    save_indices = np.array(sorted({int(np.argmin(np.abs(tau - t))) for t in times_to_save}))
    saved_times = tau[save_indices]
    nsave = len(save_indices)
    two_pi = 2*np.pi
    u_opt     = (n/c_light) * I0
    u_ac      = c_sound**2 * rho_0 * rho_bar**2 / 2
    flux_pref = 2 * I0

    def grid_energy(intensity2d, x, coeff):
        """coeff * INT intensity * 2*pi*r dr dx over the whole grid (static stored energy, no factor 2)."""
        Pz = np.trapezoid(intensity2d * rad[None, :], rad, axis=1)   # INT ()*r dr -> (Nzeta,)
        return coeff * two_pi * np.trapezoid(Pz, x)

    # on-grid energy (seed + pump + acoustic) at each save time, mapped to z_true
    E_grid = np.empty(nsave)
    for p in range(nsave):
        z_true_p = zeta - c_light*saved_times[p]/n
        E_grid[p] = (grid_energy(np.abs(b_out[p])**2, z_true_p, u_opt)
                     + grid_energy(np.abs(a_out[p])**2, z_true_p, u_opt)
                     + grid_energy(np.abs(f_out[p])**2, z_true_p, u_ac))
    grid_gain = E_grid - E_grid[0]

    # cumulative net pump flux through the moving boundaries (full tau, then sampled at save times)
    def cum_flux(col):
        P = two_pi * np.trapezoid(np.abs(col)**2 * rad[None, :], rad, axis=1)   # INT|.|^2 2pi r dr per time
        return flux_pref * np.concatenate(([0.0], np.cumsum(0.5*(P[1:]+P[:-1])*np.diff(tau))))
    E_in_cum  = cum_flux(a_in_col)
    E_out_cum = cum_flux(a_out_col)
    net_flux_in = (E_in_cum - E_out_cum)[save_indices]

    return net_flux_in, grid_gain

flux, gain = energy_change()
