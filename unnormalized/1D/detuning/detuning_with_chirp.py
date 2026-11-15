#Computing and mounting the necessary constants------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

GammaB = 2.0974e9      ###             ###                          #Hz. Brillouin Linewidth of Kr gas
OmegaB = 1.5075e9                ###            ###                 #Hz. Frequency of Brillouin Shift.
rho_0 = 3.74    ###               ###                  ###       #kg/m3. Equilibrium gas density.
n = 1.000495      ##        ##            ##            ##       #n/a. refractive index of Kr gas.
gamma = 5/3      ###                  ###                        #n/a. Adiabatic index
dlambda = 1e-12 ####                                           # for detuning laser wavelength diff

mu_0 = 4*np.pi*1e-7               ###                ###         #vacuum permeability
ep_0 = 8.854e-12          ###                    ###             #vacuum permittivity
c_light = 1/np.sqrt(mu_0 * ep_0) ###             ###             #m/s
rho_bar = 1.e-4        ###       ###          ###         ###    #n/a. small ad-hoc density perturbation scale.
gamma_e = 2*(n-1)    ###         ###            ###              #n/a. Electrostriction coefficient
P = 101325         ###                          ###              #kg m-1 s-2. Gas pressure
c_sound = np.sqrt(gamma*P/rho_0)      ###             ###        #m/s. Speed of sound in medium.
lambdaL = 248*1e-9        ###                ###                 #m. wavelength of laser light. 
fL = c_light/lambdaL      ###                   ###         #Hz. Frequency of laser light. 
wL = 2*np.pi*fL                ###               ###   #rad/s. Angular frequency of laser light. 
kL = wL/c_light                                             #rad/m. Wavenumber of laser light.
wB = 2*np.pi*OmegaB                   ###            ###           #rad/s. Angular Frequency of Brillouin Shift.
KB = wB/c_sound      ###            ###                        #rad/m. Wavenumber of acoustic wave.
I0 = 1e12    ###               ###               ###            #W/m2. Intensity of pump laser.
kNorm = rho_bar*gamma_e*wL/(4*c_light*n)                    #rad/m. normalization wavenumber.
wNorm = c_light * kNorm/n          ###             ###           #rad/s. Normalization frequency.
g = gamma_e * KB * I0 / (2 *GammaB*c_light*c_sound*rho_0*rho_bar) #rad. gain coefficient
#I_0 = c_light*ep_0*E0**2
G = GammaB / (2 * wNorm)                                            #1/rad. Normalized damping coefficient.

#Set Detuning

dlambda = 1e-12 ####                                           # for detuning laser wavelength diff
Nlinewidths = 1.5####                                            #how many G_b from f_B is Omega
#Omega = c_light/(lambdaL**2) * dlambda                         #Hz. Translated frequency difference from dlambda
Omega = OmegaB + Nlinewidths *GammaB                            #Hz. Translated frequency difference from dlambda
d = (Omega**2 - OmegaB**2)/(GammaB*Omega)                         #d from physical parameters
#d = 2                                                         #d from raw

print(f"c_light = {c_light/1e8}*10^8m/s")
print(f"c_sound = {c_sound}m/s")
print(f"g = {g}")
print(f"G = {G}")
print(f"w_laser = {wL/1e15}*10^15Hz")
print(f"wNorm = {wNorm}Hz")
print(f"k = {kNorm}")
print(f"w_B = {wB}")
print(f"d = {d}")
#Natural Parameters of detuning ------------------------------------------------------------------------------------------
true_beat = (GammaB*d + np.sqrt(GammaB**2 * d**2 + 4*OmegaB**2))/2
dlambda = lambdaL**2 / c_light * true_beat
Nlinewidths = (true_beat - OmegaB)/GammaB

print(f"Omega = {true_beat/1e9} GHz")
print(f"Dlambda = {round(dlambda*1e12,2)} pm")
print(f"Omega = {OmegaB} + {Nlinewidths} GammaB")
#Setup------------------------------------------------------------------------------------------
dzeta = 0.01                       # zeta resolution (m)
dtau = 0.8 * dzeta / c_light       # CFL-style: 0.8 * (dzeta / c). tau is TRUE time (s); c/n≈c since n≈1
tau_range = [0, 130e-9]
zeta_range = [-20, 5]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index (s)
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta), lab position (m)

a0 = 1
b0 = 1
sigma = 1.5   # chosen so pulse_duration(t=0) ≈ 0.3 m (lab-frame length metric)
phi_a_initial = 0
phi_b_initial = 0

#Set piecewise (or other) detuning across zeta------------------------------------------------
d = np.ones_like(zeta)
d[:np.argmin(np.abs(zeta + 3.5))] = 0
Omega = (GammaB*d + np.sqrt(GammaB**2 * d**2 + 4*OmegaB**2))/2


#Sample plot detuning and Omega across zeta alongside seed------------------------------------------------
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()

line1 = ax1.plot(zeta, Omega/1e9, label="Beat Frequency")
ax1.set_xlabel("Zeta (m)")
ax1.set_ylabel("Omega (GHz)")

line2 = ax2.plot(zeta, d, linestyle="--", color="red", label="Detuning Factor",  zorder=10)
ax2.set_ylabel("Detuning Factor")

line3 = ax3.plot(zeta, (b0 * np.exp(-np.abs(zeta/sigma)**2))**2, linestyle="--", color="black", label="Envelope")
ax3.set_ylabel("Envelope (Initial Intensity)")
ax3.spines['right'].set_position(('outward', 55))   # offset the 3rd axis

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, loc='upper right')

#ax1.set_xlim(-5,2)
#ax2.set_ylim(-5,5)
plt.title("Chirp Visualization")
plt.show()

#help functions:
b = 1 #placeholder
a=1
def FWHM(z_true=z_true, b=b, tau=tau):
    def _fwhm(x, y):
        if y.size < 2:
            return np.nan
        ipk  = int(np.argmax(y))
        half = 0.5 * y[ipk]
        if not np.isfinite(half) or half <= 0:
            return np.nan

        # left crossing: last node before the peak that sits below half max
        lo = np.flatnonzero(y[:ipk] < half)
        if lo.size == 0:
            return np.nan                      # peak not resolved on the left
        i = lo[-1]
        xlo = x[i] + (half - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])

        # right crossing: first node after the peak that sits below half max
        hi = np.flatnonzero(y[ipk+1:] < half)
        if hi.size == 0:
            return np.nan                      # peak not resolved on the right
        j = ipk + 1 + hi[0]
        xhi = x[j-1] + (half - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1])

        return float(xhi - xlo)

    return np.array([_fwhm(z_true[ti, :], np.abs(b[ti, :])**2) for ti in range(b.shape[0])])

#implicit efficiency is a good approximation for later times when b>>a (explicit efficiency is unused because integration overestimates seed growth)

def get_eff(a = a, tau=tau):
    a_in  = np.abs(a[:, 0])**2
    a_out = np.abs(a[:, -1])**2
    cumtrap = lambda y: np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(tau))))
    supplied = cumtrap(a_in)
    absorbed = cumtrap(a_in - a_out)
    eta = np.divide(absorbed, supplied, out=np.full_like(tau, np.nan, dtype=float), where=(supplied > 0))
    eta[0] = np.nan
    return eta

#Integrator Function--------------------------------------------------------------------------------
def detune(d=d, zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
           a0=a0, b0=b0, sigma=sigma, g=g, G=G, b_exp=b_exp,
           phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    import numpy as np
    import matplotlib.pyplot as plt
    #PARAMS####################################################
    a0 = a0
    g = g
    sigma = sigma
    b0 = b0
    ########################################################
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    Ka = kNorm/2
    Kb = kNorm*c_light/n
    kappa = 2*c_light/(GammaB*n)
    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)
    f = np.empty((Ntau, Nzeta), dtype=complex)
    #####################ICs#############################
    """
    a, f demand far-left limits.
    b requires initial conditions
    """
    a[:,0] = a0 * np.exp(1j*phi_a_initial)              # constant pump phase at entrance
    b0_profile = b0 * np.exp(-np.abs(zeta/sigma)**b_exp)
    b[0,:] = b0_profile * np.exp(1j*phi_b_initial)
    f[:,0] = 0                                          # far-left column: no incoming acoustic
             
    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] + dzeta * Ka * (-b[0,j] * f[0,j])
        f[0,j+1] = (g * a[0,j+1] * np.conj(b[0,j+1]) + kappa/dzeta * f[0,j]) / (1 + kappa/dzeta - 1j*d[j+1])

    b[1,:] = b[0,:] + dtau * Kb * (a[0,:] * np.conj(f[0,:]))

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]
            db = b[i+1,j+1] - bb
            aa = a[i+1,j]
            ff = f[i+1,j]
            if 2*G>1:
                a[i+1,j+1] = aa + dzeta * Ka * (-bb * ff)
                f[i+1,j+1] = (g * a[i+1,j+1] * np.conj(b[i+1,j+1]) + kappa/dzeta * f[i+1,j]) / (1 + kappa/dzeta - 1j*d[j+1])
            else:
                da1 = dzeta * Ka * (-bb * ff)
                df1 = dzeta/kappa * (g * aa * np.conj(bb) - (1 - 1j*d[j+1]) * ff)

                da2 = dzeta * Ka * (-(bb + db/2) * (ff + df1/2))
                df2 = dzeta/kappa * (g * (aa + da1/2) * np.conj(bb + db/2) - (1 - 1j*d[j+1]) * (ff + df1/2))

                da3 = dzeta * Ka * (-(bb + db/2) * (ff + df2/2))
                df3 = dzeta/kappa * (g * (aa + da2/2) * np.conj(bb + db/2) - (1 - 1j*d[j+1]) * (ff + df2/2))

                da4 = dzeta * Ka * (-(bb + db) * (ff + df3))
                df4 = dzeta/kappa * (g * (aa + da3) * np.conj(bb + db) - (1 - 1j*d[j+1]) * (ff + df3))

                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6

        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * (a[i+1,:] * np.conj(f[i+1,:]))

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    phi_f = np.angle(f)

    print(a.shape)
    print(b.shape)
    print(np.any(np.isinf(b)))
    print(np.any(np.isnan(b)))
    # find where blowup first occurs
    blown = np.where(np.isinf(b))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    return a, b, f, phi_a, phi_b, phi_f

#Example Usage

sigmas       = [0.5,0.6,0.7,0.8]
b_0_initials = [1]
DETUNING     = np.zeros_like(zeta)

runs = {}   # key: (b0, sigma) -> dict of arrays
for b0 in b_0_initials:
    for s in sigmas:
        a, b, f, *_ = detune(d=DETUNING, sigma=s, b0=b0)   # CHECK kwarg name
        runs[(b0, s)] = dict(a=a, b=b, f=f, fwhm=FWHM(b=b))
        print(f"done  b0 = {b0:<5}  sigma = {s}")

#Plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- toggles ----------------
SHOW_SEED   = True
SHOW_PUMP   = True
SHOW_AC     = True
SHOW_TARGET = True
SHOW_ANNOT  = True
SHOW_LEGEND = True
USE_XLIM    = False

# ---------------- settings ----------------
TIMES      = [800e-9]             # snapshot times (s) -- add as many as you like
COLORS     = ["blue", "green", "blue"]   # used in order; extra snapshots fall back to tab10
ANNOT_IDX  = -1                         # which snapshot the annotation reports (-1 = last)
XLIM       = (-2.5, 1.5)
FWHM_UNIT  = "m"
TARGET     = -0.7
PARAMS = {r"$g$": g, r"$\Gamma_B$": GammaB,
          r"$n$": n, r"$k_{norm}$": kNorm}

# ---------------- unit conversions ----------------
field_norm = (I0/2*1e-10) / 1000.0                   # |b|^2 -> MW/cm^2 -> GW/cm^2
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm    = 1000.0                                  # J/m^3 -> mJ/m^3

# ---------------- grid ----------------
colors   = [COLORS[k] if k < len(COLORS) else plt.cm.tab10(k % 10)
            for k in range(len(TIMES))]
t_idxs   = [np.argmin(np.abs(tau - T)) for T in TIMES]
t_actual = [tau[i] for i in t_idxs]
t_idx    = t_idxs[ANNOT_IDX]            # used by pick() and the annotation
pick     = lambda q: np.asarray(q)[t_idx] if np.ndim(q) else q
i_target = np.argmin(np.abs(zeta - TARGET))

nrow, ncol = len(b_0_initials), len(sigmas)
fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey="row",
                         figsize=(6.0 * ncol, 5 * nrow), squeeze=False)

ac_axes   = np.empty((nrow, ncol), dtype=object)
pump_axes = np.empty((nrow, ncol), dtype=object)

for r, b0 in enumerate(b_0_initials):
    for j, s in enumerate(sigmas):
        run = runs[(b0, s)]
        ax  = axes[r, j]

        I_snaps = [np.abs(run["b"][ti, :])**2 * field_norm for ti in t_idxs]

        # ---- seed ----
        if SHOW_SEED:
            for I_k, col in zip(I_snaps, colors):
                ax.plot(zeta, I_k, color=col, linestyle="-")

        if SHOW_TARGET:
            ax.axvline(TARGET, color="black", linestyle="--", alpha=0.5)

        # ---- pump on its own axis ----
        if SHOW_PUMP:
            ax_pump = ax.twinx()
            pump_axes[r, j] = ax_pump
            if j > 0:                              # share within the row
                ax_pump.sharey(pump_axes[r, 0])
            for ti, col in zip(t_idxs, colors):
               ax_pump.plot(zeta, np.abs(run["a"][ti, :])**2 * field_norm,
                             color=col, linestyle="--", alpha=0.8)
               ax_pump.set_ylim(top = 3*np.max(np.abs(run["a"][ti, :])**2 * field_norm))
            if j == ncol - 1:
                ax_pump.set_ylabel(r"Pump (GW/cm$^2$)")
            else:
                ax_pump.tick_params(labelright=False)

        # ---- acoustic energy density on its own axis ----
        if SHOW_AC:
            ax_ac = ax.twinx()
            ac_axes[r, j] = ax_ac
            if SHOW_PUMP:                          # keep clear of the pump axis
                ax_ac.spines['right'].set_position(('outward', 55))
            if j > 0:                              # share within the row
                ax_ac.sharey(ac_axes[r, 0])
            for ti, col in zip(t_idxs, colors):
                ax_ac.plot(zeta, np.abs(run["f"][ti, :])**2 * u_ac * ac_norm,
                           color=col, linestyle=":", alpha=0.8)
            if j == ncol - 1:
                ax_ac.set_ylabel(r"Acoustic (mJ/m$^3$)")
            else:
                ax_ac.tick_params(labelright=False)


        # ---- annotation box ----
        if SHOW_ANNOT:
            I_ann = I_snaps[ANNOT_IDX]
            ax.text(0.03, 0.95,
                    f"peak = {I_ann.max():.3g} GW/cm$^2$\n"
                    f"FWHM = {pick(run['fwhm']):.3g} {FWHM_UNIT}\n"
                    f"TARGET = {I_ann[i_target]:.3g} GW/cm$^2$",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, zorder=5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

        # snapshot-time key, on every panel
        for k, (t, col) in enumerate(zip(t_actual, colors)):
            ax.text(0.97, 0.95 - 0.07 * k, f"t = {t*1e9:.0f} ns",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=col, fontweight="bold", zorder=5)

        if r == 0:
            ax.set_title(rf"$\sigma$ = {s}m")
        if r == nrow - 1:
            ax.set_xlabel(r"$\zeta$ (m)")
        ax.grid(alpha=0.3)

    axes[r, 0].set_ylabel(f"$b_0$ = {b0} {b0*field_norm} GW/cm$^2$" + ("\nSeed (GW/cm$^2$)" if SHOW_SEED else ""))
    if not SHOW_SEED:
        axes[r, 0].tick_params(labelleft=False)    # empty seed axis: no misleading ticks

if USE_XLIM:
    axes[0, 0].set_xlim(*XLIM)

if SHOW_LEGEND:
    handles = []
    if SHOW_SEED:
        handles.append(Line2D([0], [0], color="gray", linestyle="-",  label="Seed"))
    if SHOW_PUMP:
        handles.append(Line2D([0], [0], color="gray", linestyle="--", label="Pump"))
    if SHOW_AC:
        handles.append(Line2D([0], [0], color="gray", linestyle=":",  label="Acoustic"))
    if handles:
        axes[0, 0].legend(handles=handles, loc="lower right", fontsize=8)

title = ", ".join(f"{k} = {v:.3g}" for k, v in PARAMS.items())
if SHOW_ANNOT:
    title += f"   |   annotations at t = {t_actual[ANNOT_IDX]*1e9:.1f} ns"
fig.suptitle(title, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
