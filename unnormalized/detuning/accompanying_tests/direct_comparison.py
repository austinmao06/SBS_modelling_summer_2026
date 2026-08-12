#Run different detuning factors-----------------------------------------------------------------------------

detuning_factors = [0, d/5, 2*d/5, 3*d/5,4*d/5, d]
nruns = len(detuning_factors)
a_runs, b_runs, f_runs = [], [], []
for i, dval in enumerate(detuning_factors):
    a_i, b_i, f_i, *_ = detune(d=dval)
    a_runs.append(a_i); b_runs.append(b_i); f_runs.append(f_i)
    globals()[f"a{i}"], globals()[f"b{i}"], globals()[f"f{i}"] = a_i, b_i, f_i

#compare SBS amplification at different times (with physical units)--------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

I_scale    = 100
field_norm = I_scale / 1000.0
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0
ac_norm    = 1000.0

PUMP_SCALE = 0.05
AC_SCALE   = 0.25

times    = [40e-9, 80e-9, 126.7e-9]
# one row per detuning run (auto-adjusts to len(detuning_factors))
datasets = [(b_runs[i], a_runs[i], f_runs[i], detuning_factors[i]) for i in range(nruns)]

nrows, ncols = len(datasets), len(times)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                         figsize=(16, 3.2 * nrows), squeeze=False)

ax_pump_ref = ax_ac_ref = None
pump_axes, ac_axes, seed_lines = [], [], []

for row, (bb, aa, ff, delta) in enumerate(datasets):
    for i, t in enumerate(times):
        t_idx = np.argmin(np.abs(tau - t))
        ax = axes[row, i]

        ax_pump = ax.twinx()
        ax_ac   = ax.twinx()
        ax_ac.spines['right'].set_position(('outward', 55))

        if ax_pump_ref is None:
            ax_pump_ref, ax_ac_ref = ax_pump, ax_ac
        else:
            ax_pump.sharey(ax_pump_ref)
            ax_ac.sharey(ax_ac_ref)

        line_seed, = ax.plot(zeta, np.abs(bb[t_idx, :])**2 * field_norm,
                             color="black", linestyle="-")
        ax_pump.plot(zeta, np.abs(aa[t_idx, :])**2 * field_norm,
                     color="black", linestyle="--")
        ax_ac.plot(zeta, np.abs(ff[t_idx, :])**2 * u_ac * ac_norm,
                   color="black", linestyle=":")

        pump_axes.append(ax_pump)
        ac_axes.append(ax_ac)

        if row == 0:
            ax.set_title(f"t = {round(t*1e9, 1)} ns")
        if row == nrows - 1:
            ax.set_xlabel(r"$\zeta$ (m)")
        if i < ncols - 1:
            ax_pump.tick_params(labelright=False)
            ax_ac.tick_params(labelright=False)
            ax_ac.spines['right'].set_visible(False)
        if i == 0:
            seed_lines.append((line_seed, delta))

        ax.grid(alpha=0.3)

# ---------------- scaling ----------------
axes[0, 0].autoscale(axis='y')
y0, y1 = axes[0, 0].get_ylim()
ax_pump_ref.set_ylim(y0 * PUMP_SCALE, y1 * PUMP_SCALE)
ax_ac_ref.set_ylim(y0 * AC_SCALE,     y1 * AC_SCALE)

# ---------------- labels & legends ----------------
style_handles = [
    Line2D([0], [0], color="gray", linestyle="-",  label="Seed"),
    Line2D([0], [0], color="gray", linestyle="--", label="Pump"),
    Line2D([0], [0], color="gray", linestyle=":",  label="Acoustic"),
]

for row in range(nrows):
    axes[row, 0].set_ylabel(r"Seed intensity (GW/cm$^2$)")
    k = row * ncols + (ncols - 1)                     # last column of this row
    pump_axes[k].set_ylabel(r"Pump intensity (GW/cm$^2$)")
    ac_axes[k].set_ylabel(r"Acoustic energy density (mJ/m$^3$)")

    line_seed, delta = seed_lines[row]
    axes[row, 0].legend([line_seed] + style_handles,
                        [f"δ = {delta}", "Seed", "Pump", "Acoustic"],
                        loc="upper right", fontsize=8)

axes[0, 0].set_xlim(-1, 2)
fig.tight_layout()
plt.show()

#Alternative: Direct overlay---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

I_scale    = 100
field_norm = I_scale / 1000.0
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0
ac_norm    = 1000.0

PUMP_SCALE = 0.05
AC_SCALE   = 0.25

times       = [40e-9, 80e-9, 126.7e-9]
time_colors = ["red", "green", "blue"]

# one row per detuning run
datasets = [(b_runs[i], a_runs[i], f_runs[i], detuning_factors[i])
            for i in range(nruns)]
nrows = len(datasets)

fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=True,
                         figsize=(9, 3.2 * nrows), squeeze=False)
axes = axes[:, 0]                      # flatten to 1-D

ax_pump_ref = ax_ac_ref = None
pump_axes, ac_axes = [], []

for row, (bb, aa, ff, delta) in enumerate(datasets):
    ax = axes[row]

    ax_pump = ax.twinx()
    ax_ac   = ax.twinx()
    ax_ac.spines['right'].set_position(('outward', 55))

    if ax_pump_ref is None:
        ax_pump_ref, ax_ac_ref = ax_pump, ax_ac
    else:
        ax_pump.sharey(ax_pump_ref)
        ax_ac.sharey(ax_ac_ref)

    # all three times overlaid on the same axes, distinguished by color
    for t, color in zip(times, time_colors):
        t_idx = np.argmin(np.abs(tau - t))
        ax.plot(zeta, np.abs(bb[t_idx, :])**2 * field_norm,
                color=color, linestyle="-")
        ax_pump.plot(zeta, np.abs(aa[t_idx, :])**2 * field_norm,
                     color=color, linestyle="--")
        ax_ac.plot(zeta, np.abs(ff[t_idx, :])**2 * u_ac * ac_norm,
                   color=color, linestyle=":")

    pump_axes.append(ax_pump)
    ac_axes.append(ax_ac)

    ax.set_ylabel(r"Seed intensity (GW/cm$^2$)")
    ax_pump.set_ylabel(r"Pump intensity (GW/cm$^2$)")
    ax_ac.set_ylabel(r"Acoustic (mJ/m$^3$)")
    ax.set_title(rf"$\delta$ = {delta}", loc="left", fontsize=10)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel(r"$\zeta$ (m)")

# ---------------- scaling ----------------
axes[0].autoscale(axis='y')
y0, y1 = axes[0].get_ylim()
ax_pump_ref.set_ylim(y0 * PUMP_SCALE, y1 * PUMP_SCALE)
ax_ac_ref.set_ylim(y0 * AC_SCALE,     y1 * AC_SCALE)

# ---------------- legend ----------------
time_handles = [Line2D([0], [0], color=c, linestyle="-",
                       label=f"{t*1e9:.0f} ns")
                for t, c in zip(times, time_colors)]
style_handles = [
    Line2D([0], [0], color="gray", linestyle="-",  label="Seed"),
    Line2D([0], [0], color="gray", linestyle="--", label="Pump"),
    Line2D([0], [0], color="gray", linestyle=":",  label="Acoustic"),
]
axes[0].legend(handles=time_handles + style_handles,
               loc="upper right", fontsize=8, ncol=2)

axes[0].set_xlim(-4, 1)
fig.tight_layout()
plt.show()
