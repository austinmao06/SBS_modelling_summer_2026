#Leading edge coordinate of each radius at every saved time. Used for parallel run leading edge analysis

def leading_edge_2D(b, zeta=zeta, frac=0.5):
    I = np.abs(b)**2                                   # (Ntime, Nzeta, Nrad)
    Ntime, Nzeta, Nrad = I.shape
    level = frac * I[0].max(axis=0)                     # per-radius initial-peak level -> (Nrad,)

    def _leading_edge(x, y, lvl):
        """zeta of the lowest-zeta (leading-edge) up-crossing of `lvl`, interpolated; NaN if none."""
        if not np.isfinite(lvl) or lvl <= 0:
            return np.nan
        pk = int(np.argmax(y))
        for i in range(pk, 0, -1):                     # scan from the peak toward lower zeta
            y0, y1 = y[i-1], y[i]
            if y0 <= lvl <= y1:                        # rising crossing between i-1 and i
                return x[i-1] if y1 == y0 else x[i-1] + (lvl - y0) * (x[i] - x[i-1]) / (y1 - y0)
        return np.nan

    out = np.full((Ntime, Nrad), np.nan)
    for ti in range(Ntime):
        for r in range(Nrad):
            out[ti, r] = _leading_edge(zeta, I[ti, :, r], level[r])
    return out
