#Obtains time at which seed leading edge has travelled dist_true m
def get_time(dist_true, z_true = z_true, intensity = b**2, t_true = t_true):
    threshold = intensity[0, :].max() #Leading edge point is defined as where intensity = initial MAX intensity
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
#Example usage
t = get_time(38)
print(f"The seed travels 38m in {t*1e9}ns")
