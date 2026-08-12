#Tracks zeta-coordinate of a fixed intensity level (fraction of initial seed) over time
def leading_edge(b=b, zeta=zeta, tau=tau, frac=0.5):
    I = np.abs(b)**2
    level = frac * I[0, :].max()                       # fixed threshold from the initial profile

    def _leading_edge(x, y, lvl):
        """zeta of the lowest-zeta (leading-edge) up-crossing of `lvl`, interpolated; NaN if none."""
        pk = int(np.argmax(y))
        for i in range(pk, 0, -1):                     # scan from the peak toward lower zeta
            y0, y1 = y[i-1], y[i]
            if y0 <= lvl <= y1:                        # rising crossing between i-1 and i
                return x[i-1] if y1 == y0 else x[i-1] + (lvl - y0) * (x[i] - x[i-1]) / (y1 - y0)
        return np.nan

    return np.array([_leading_edge(zeta, I[ti, :], level) for ti in range(len(tau))])
#Example usage

edge = leading_edge()
plt.plot(tau,edge, label="Leading Edge coordinate over Time")
