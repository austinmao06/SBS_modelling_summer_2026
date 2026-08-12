##Returns time for pulse leading edge to have moved dist_true m

def get_time_at(dist_true):
    intensity = b_axis**2
    threshold = intensity[0, :].max()
    Ntau = intensity.shape[0]

    def leading_z(ti):
        """z of the leading-edge (more-negative-z) threshold crossing at row ti."""
        y = intensity[ti, :]
        x = z_true[ti, :]
        pk = np.argmax(y)
        # scan from peak toward more-negative z (lower index)
        for i in range(pk, 0, -1):
            if y[i] >= threshold >= y[i-1]:
                return x[i-1] + (threshold - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
        return np.nan

    z0 = leading_z(0)                                   # leading-edge position at t=0
    disp = np.array([z0 - leading_z(ti) for ti in range(Ntau)])   # displacement toward -z

    # first time displacement reaches dist_true, interpolated in time
    for ti in range(1, Ntau):
        if disp[ti] >= dist_true >= disp[ti-1]:
            frac = (dist_true - disp[ti-1]) / (disp[ti] - disp[ti-1])
            return t_true[ti-1] + frac * (t_true[ti] - t_true[ti-1])
    return np.nan
