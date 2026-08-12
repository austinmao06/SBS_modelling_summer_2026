#Plot Seed Frame with proper scaling----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

radius_of_interest = 1.9e-2
r = np.argmin(np.abs(rad - radius_of_interest))

# --- unit conversions ---------------------------------
field_norm = 100 / 1000          # your existing |a|^2,|b|^2 -> GW/cm^2 factor
                          # (replace with the actual E0-based factor if different)
u_ac  = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm = 1.0 * 1000

PUMP_SCALE = 0.05
AC_SCALE = 0.1

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()                      # pump, same optical units
ax3 = ax1.twinx()                      # acoustic, its own units
ax3.spines['right'].set_position(('outward', 55))   # offset the 3rd axis

def plot_set(t_idx, color, label):
    ax1.plot(zeta, np.abs(b[t_idx][:,r])**2 * field_norm, color=color,
             linestyle="-",  label=label)
    ax2.plot(zeta, np.abs(a[t_idx][:,r])**2 * field_norm, color=color, linestyle="--")
    ax3.plot(zeta, np.abs(f[t_idx][:,r])**2 * u_ac * ac_norm,  color=color, linestyle=":")

TIMES = [40e-9, 80e-9, 126.7e-9]
COLORS = ["red", "green", "blue"]

for i, ti in enumerate(TIMES):
  time_i = np.argmin(np.abs(tau-ti))
  plot_set(time_i, COLORS[i], f"{ti*1e9}ns")


ax1.set_xlabel(r"$\zeta$ (m)")
ax1.set_ylabel(r"Seed intensity (GW/cm$^2$)")
ax2.set_ylabel(r"Pump intensity (GW/cm$^2$)")
ax3.set_ylabel(r"Acoustic energy density (mJ/m$^3$)")

style_legend = [
    Line2D([0], [0], color="black", linestyle="-",  label=r"seed"),
    Line2D([0], [0], color="black", linestyle="--", label=r"pump"),
    Line2D([0], [0], color="black", linestyle=":",  label=r"acoustic"),
]
color_legend = ax1.get_legend_handles_labels()[0]
ax1.legend(handles=color_legend + style_legend, loc="best")

ax1.set_xlim(-10, 1)
seed_min, seed_max = ax1.get_ylim()
ax2.set_ylim(seed_min * PUMP_SCALE, seed_max * PUMP_SCALE)           
ax3.set_ylim(seed_min * AC_SCALE, seed_max * AC_SCALE)  

plt.title("Finite Damping Amplification (Gaussian Seed Profile)")
plt.tight_layout()
ax1.grid()
plt.show()
