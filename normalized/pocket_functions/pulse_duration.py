"""Pulse Duration defined as the shortest true distance over which 80% of total seed energy is contained
Returns pulse duration at each time"""

def pulse_duration(z_true=z_true, b=b, tau=tau, frac=0.8):
    def _dur(x, y):
        if y.max() <= 0:
            return np.nan
        C = np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))))   # cumulative area
        total = C[-1]
        if total <= 0:
            return np.nan
        zpk = x[int(np.argmax(y))]                                    # absolute-peak position
        zhi = np.interp(C + frac*total, C, x, right=np.nan)           # right edge of frac-area window per left node
        length = zhi - x
        m = (x <= zpk) & (zhi >= zpk) & np.isfinite(zhi) & (length > 0)   # window must contain the peak
        return float(np.min(length[m])) if np.any(m) else np.nan
    return np.array([_dur(z_true[ti, :], b[ti, :]**2) for ti in range(len(tau))])
#Example usage
dur = pulse_duration()
plt.plot(time_true*1e9, pulse_duration)
plt.xlabel("Time Elapsed (ns)")
plt.ylabel("Pulse Duration (m)")
