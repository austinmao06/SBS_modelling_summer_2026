#Min distance over which 80% of seed energy is contained

def seed_width(b_out = b, times=times_to_save, zeta=zeta, rad=rad, frac=0.8):
    save_indices = sorted({int(np.argmin(np.abs(tau - t))) for t in times})
    saved_times = tau[save_indices]

    def _dur(x, y):
        if y.max() <= 0:
            return np.nan
        C = np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))))   # cumulative area
        total = C[-1]
        if total <= 0:
            return np.nan
        zpk = x[int(np.argmax(y))]                                    # absolute-peak position
        zhi = np.interp(C + frac*total, C, x, right=np.nan)           # right edge of frac-area window
        length = zhi - x
        m = (x <= zpk) & (zhi >= zpk) & np.isfinite(zhi) & (length > 0)   # window must contain the peak
        return float(np.min(length[m])) if np.any(m) else np.nan

    out = np.empty(len(saved_times))
    for p in range(len(saved_times)):
        z   = zeta - c_light*saved_times[p]/n                         # z_true at this slice (r-independent)
        P_b = np.trapezoid(b_out[p]**2 * rad[None, :], rad, axis=1)   # seed power per zeta
        out[p] = _dur(z, P_b)
    return out

#Example Usage

w = seed_width()
plt.plot(times_to_save, w)
