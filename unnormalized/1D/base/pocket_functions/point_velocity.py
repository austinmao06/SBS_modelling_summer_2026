#Returns zeta values for multiple intensity levels of the seed 
def point_velocity(b=b, zeta=zeta, tau=tau, fracs=(0.2, 0.4, 0.6, 0.8, 1.0)):
    fracs = np.asarray(fracs, dtype=float)
    I = np.abs(b)**2

    def _leading_edge(x, y, lvl):
        """zeta of the lowest-zeta (leading-edge) up-crossing of `lvl`, interpolated; NaN if none."""
        pk = int(np.argmax(y))
        for i in range(pk, 0, -1):                     # scan from the peak toward lower zeta
            y0, y1 = y[i-1], y[i]
            if y0 <= lvl <= y1:                        # rising crossing between i-1 and i
                return x[i-1] if y1 == y0 else x[i-1] + (lvl - y0) * (x[i] - x[i-1]) / (y1 - y0)
        return np.nan

    out = np.full((len(tau), len(fracs)), np.nan)
    for ti in range(len(tau)):
        y = I[ti, :]
        ymax = y.max()
        if ymax <= 0:
            continue
        for fi, frac in enumerate(fracs):
            out[ti, fi] = _leading_edge(zeta, y, frac * ymax)
    return out

#example usage
point_vs = point_velocity()
plt.plot(tau, point_vs[:,0], label="20% of peak leading edge coordinate")

