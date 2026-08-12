#Plots full snapshot of a, b, f

t = 5

imgs = []
fig, axarr = plt.subplots(3,1, figsize=(20, 10))
labels = [r"Seed Intensity (GW/cm$^2$)",
          r"Pump Intensity (GW/cm$^2$)",
          r"Acoustic (mJ/m$^3$)"]
multipliers = [0.1, 0.1,c_sound**2 * rho_0 * rho_bar**2 / 2.0 * 1000]
axylims = [[0,0.0075],[0,0.0075],[0,0.0075]]

for ax, field, lab, m, lims in zip(axarr, (a,b, f), labels, multipliers, axylims):   # note order: seed, pump, acoustic
    I = (field[t, :, :]**2).T * m                      # shape (Nrad, Nzeta)
    im = ax.imshow(I, aspect="auto", origin="lower",
                   extent=[zeta[0], zeta[-1], rad[0], rad[-1]],
                   cmap="inferno")
    ax.set_xlabel(r"$\zeta$ (m)")
    ax.set_ylabel("Radius (m)")
    fig.colorbar(im, ax=ax, label=lab)
    ax.set_ylim(lims[0], lims[1])
    ax.axhline(0.005, linestyle="--", color="white")


fig.suptitle(f"Intensity at Different Radii (t = {round(times_to_save[t]*1e9)}ns)")
ax1.set_title("Pump")
ax2.set_title("Seed")
ax3.set_title("Acoustic")
plt.tight_layout()
plt.show()
