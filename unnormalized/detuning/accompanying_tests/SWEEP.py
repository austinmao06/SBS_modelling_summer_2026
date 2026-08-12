#Sweep detuning factors-----------------------------------------------------------------------------
ds = np.arange(0,10, 0.3)
bs = [None] * len(ds)
a_s = [None] * len(ds)
for i in range(len(ds)):
    a_s[i], bs[i], *_ = detune(d=ds[i])
a_ref, b_ref, *_ = detune(d=0)

#Compute peak intensity, efficiency, pulse duration for each at different timesteps-----
peaks = np.array([np.max(bs[i]**2, axis = 1) / 10 for i in range(len(ds))])
peaks_ref = np.max(b_ref**2, axis = 1) / 10
effs     = np.array([get_eff(a) for a in a_s])   # pump arrays per detuning
eff_ref  = get_eff(a_ref)
durs    = np.array([pulse_duration(b=b) for b in bs]) *n/c_light *1e9
dur_ref = pulse_duration(b=b_ref) *n/c_light *1e9
#Plot on 3x3 grid. Each column -> different time. Each row-> peak I/eff/dur-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

times_ns = [40, 80, 126.7]                                          # column times (ns)
time_idx = [int(np.argmin(np.abs(tau - t*1e-9))) for t in times_ns]  # ns -> tau index (tau is true time in s)

rows = [
    ("Peak Intensity vs Detuning", "I (GW/cm$^2$)", "C0",
     lambda ti: (peaks[:,ti],
                 peaks_ref[ti])),
    ("Efficiency vs Detuning", "Efficiency", "red",
     lambda ti: (effs[:, ti], eff_ref[ti])),
    ("Pulse Duration vs Detuning", "Pulse Duration (ns)", "green",
     lambda ti: (durs[:, ti], dur_ref[ti])),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for r, (row_title, ylabel, color, getter) in enumerate(rows):
    for c, ti in enumerate(time_idx):
        ax = axes[r, c]
        yvals, yref = getter(ti)
        ax.axhline(yref, color="black", linestyle="--", label="No Detuning Reference")
        ax.plot(ds, yvals, color=color)
        ax.scatter(ds, yvals, color=color, s = 6)
        ax.set_xlim(0, 10)
        if r == 0:
            ax.set_title(f"t = {times_ns[c]} ns", fontsize=13)
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == 2:
            ax.set_xlabel("Detuning Factor")
            ax.axhline(1, linestyle=":", color="black", label="Unamplified Reference")
axes[2,2].legend(loc="upper right", fontsize=8)
plt.tight_layout(rect=[0.05, 0, 1, 1])
for r, (row_title, *_) in enumerate(rows):                             # row titles in the left margin
    pos = axes[r, 0].get_position()
    fig.text(0.015, (pos.y0 + pos.y1)/2, row_title, rotation=90,
             va="center", ha="center", fontweight="bold", fontsize=12)
plt.show()

#Saving data if needed-----------------------------------------------------------------------------
import pandas as pd
t = 126.7e-9
time = np.argmin(np.abs(tau - t))
df = pd.DataFrame({"Detune Factor": ds, "Peak": peaks[:,time], "Eff": effs[:,time], "Dur": durs[:,time]})
print(df)
df.to_csv("Xe_sweep.txt", sep = " ", index = False)
