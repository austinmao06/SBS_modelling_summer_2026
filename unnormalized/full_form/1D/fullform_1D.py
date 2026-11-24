#gas parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import dia_array

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
dlambda = 1e-12 ####                                           # for detuning laser wavelength diff
Nlinewidths = 1####                                            #how many G_b from f_B is Omega
#Omega = c_light/(lambdaL**2) * dlambda                         #Hz. Translated frequency difference from dlambda
Omega = OmegaB + Nlinewidths *GammaB                            #Hz. Translated frequency difference from linewidths
d = (Omega**2 - OmegaB**2)/(GammaB*Omega)                         #d from physical parameters

print(f"c_light = {round(c_light/1e8,1)}*10^8m/s")
print(f"c_sound = {round(c_sound,1)}m/s")
print(f"g = {round(g,2)}")
print(f"G = {round(G,2)}")
print(f"k = {round(kNorm,4)}")
print(f"d = {round(d,3)}")


#setup
#############################################################################

#BASE SETUP##################################################################
dzeta = 0.01                       # zeta resolution (m)
dtau = 0.8 * dzeta / c_light       # CFL-style: 0.8 * (dzeta / c). tau is TRUE time (s); c/n≈c since n≈1

tau_range = [0, 130e-9]
zeta_range = [-20, 5]

tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)

Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                        # physical time per tau index (s)
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta), lab position (m)
#############################################################################

#compressed dimensions for saving###############################################################
compressed_tau = np.arange(0,tau_range[1], 1e-9)
compressed_tau_indices = sorted({int(np.argmin(np.abs(tau - t))) for t in compressed_tau})
Ncompressed_tau = len(compressed_tau_indices)
#############################################################################

#Longitudinal Pulse Parameters ##############################################
a0 = 1
b0 = 1
b_exp = 1
sigma = 1 #seed longitudinal gaussian width
phi_a_initial = 0 #initial pump phase
phi_b_initial = 0 #initial seed phase
#############################################################################

#Detuning Grid###############################################################
d = np.zeros_like(zeta) #zeta-dependent detuning factor
plt.figure(figsize=(4,3)); plt.plot(zeta, d); plt.xlabel("ζ (m)"); plt.ylabel("δ Detuning"); plt.show()
#############################################################################

#Tableau Parameters##########################################################
theta=1-1/np.sqrt(2)
eta = 1-1/(2*theta)

Tab_L = np.array([[0,0,0],[0,theta,0], [0,1-theta,theta]])
#Tab_L_weight = np.array([0,1-theta,theta])
Tab_N = np.array([[0,0,0],[theta,0,0],[eta,1-eta,0]])
#Tab_N_weight = np.array([eta,1-eta,0])

#integrator

def full_form(d=d, zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
           a0=a0, b0=b0, sigma=sigma, g=g, G=G, b_exp=b_exp,
           phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial, compressed_tau=compressed_tau,
              compressed_tau_indices = compressed_tau_indices, Ncompressed_tau = Ncompressed_tau, area = 1.0):
    #_______________________________________________________________________________________________________________
    #_______________________________________________________________________________________________________________
    #SETUP__________________________________________________________________________________________________________
    """
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)
    """
    #_________________________________________________________________________________________________________________
    #_________________________________________________________________________________________________________________
    #ICs/BCs__________________________________________________________________________________________________________
    a = np.empty((Ncompressed_tau, Nzeta), dtype=complex)
    a[0,:] = a0 * np.exp(1j*phi_a_initial)
    b = np.empty((Ncompressed_tau, Nzeta), dtype=complex)
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp) * np.exp(1j*phi_b_initial)
    f = np.empty((Ncompressed_tau, Nzeta), dtype=complex)
    f[0,:] = 0
    #_______________________________________________________________________________________________________________
    #_______________________________________________________________________________________________________________
    #DEFINE OPERATORS_______________________________________________________________________________________________
    def N(V):
        a_sub = -c_light * kNorm/n * V[Nzeta:2*Nzeta] * V[2*Nzeta:3*Nzeta]
        b_sub = c_light * kNorm/n * V[:Nzeta] * np.conj(V[2*Nzeta:3*Nzeta])
        f_sub = GammaB/2 * g * V[:Nzeta] * np.conj(V[Nzeta:2*Nzeta])
        return np.concatenate((a_sub, b_sub, f_sub))
    def BC(V):
        V[0] = a0 * np.exp(1j*phi_a_initial) #undepleted pump
        V[2*Nzeta] = 0 #no acoustic
        return V
    #Construct L Matrix______________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #zeta-transport: second-order upwind, -v dV/dzeta ~ -v (3V_i - 4V_{i-1} + V_{i-2}) / (2 dzeta)
    #(central differences give zero slope on a 2-cell checkerboard, so grid-scale seed structure gets full SBS gain)
    #row 0  : inflow BC row (zero; BC() sets a = a0, f = 0)
    #row 1  : first-order upwind -v (V_1 - V_0) / dzeta  (no V_{-1} for the 2nd-order stencil)
    #rows 2+: second-order upwind, including the outflow row (upwind needs no downstream point)
    #D0 = coefficient of V_i, D1 = of V_{i-1}, D2 = of V_{i-2}
    def upwind_bands(v, extra):
        D0 = np.full(Nzeta, -3*v/(2*dzeta), dtype=complex) + extra
        D1 = np.full(Nzeta,  4*v/(2*dzeta), dtype=complex)
        D2 = np.full(Nzeta,   -v/(2*dzeta), dtype=complex)
        D0[1] = -v/dzeta + extra[1]; D1[1] = v/dzeta; D2[1] = 0     # D2[1] = 0 also keeps the stencil inside each block
        D0[0] = 0; D1[0] = 0; D2[0] = 0
        return D0, D1, D2

    zero = np.zeros(Nzeta, dtype=complex)
    aD0, aD1, aD2 = upwind_bands(2*c_light/n, zero)                         # pump: v = 2c/n
    bD0, bD1, bD2 = zero, zero, zero                                        # seed: no zeta-transport
    fD0, fD1, fD2 = upwind_bands(c_light/n, -GammaB * (1-(1j)*d)/2)         # acoustic: v = c/n, plus damping/detuning

    D0 = np.concatenate((aD0, bD0, fD0)); D1 = np.concatenate((aD1, bD1, fD1)); D2 = np.concatenate((aD2, bD2, fD2))
    LM = np.zeros((4, 3*Nzeta), dtype = complex)                            # banded storage: rows = upper, main, lower 1, lower 2
    LM[1,:] = D0; LM[2,:-1] = D1[1:]; LM[3,:-2] = D2[2:]

    solve_M = -dtau*theta*LM
    solve_M[1, :] = solve_M[1, :] + 1
    L = dia_array((LM, [1,0,-1,-2]), shape=(3*Nzeta, 3*Nzeta))
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #Energy diagnostics setup (area in m^2)__________________________________________________________________________
    I_unit    = I0/2                                   # W/m^2 per |field|^2
    eps_light = area * (n/c_light) * I_unit            # J per (|a|^2 or |b|^2) per m of zeta
    eps_ac    = area * c_sound**2*rho_0*rho_bar**2/2   # J per |f|^2 per m of zeta
    v_pump    = 2*c_light/n                            # pump speed across the zeta boundaries
    trap      = lambda y: dzeta*(y.sum() - 0.5*(y[0] + y[-1]))
    def grid_energy(V):                                # J on the grid: pump, seed, acoustic
        return (eps_light*trap(np.abs(V[:Nzeta])**2), eps_light*trap(np.abs(V[Nzeta:2*Nzeta])**2),
                eps_ac*trap(np.abs(V[2*Nzeta:3*Nzeta])**2))
    def edge_power(V):                                 # W: pump entering at left, leaving at right = (2c/n) u A
        return v_pump*eps_light*np.abs(V[0])**2, v_pump*eps_light*np.abs(V[Nzeta-1])**2
    rec = np.empty((Ncompressed_tau, 5))                  # per saved time: pump_grid, seed_grid, ac_grid, pump_in, pump_out
    #FWHM of seed intensity |b|^2 (in zeta; identical to lab-z FWHM since z = zeta - c*tau/n is a shift)______________
    def _fwhm(x, y):
        if y.size < 2:
            return np.nan
        ipk  = int(np.argmax(y))
        half = 0.5 * y[ipk]
        if not np.isfinite(half) or half <= 0:
            return np.nan
        lo = np.flatnonzero(y[:ipk] < half)            # left crossing: last node before peak below half max
        if lo.size == 0:
            return np.nan                              # peak not resolved on the left
        i = lo[-1]
        xlo = x[i] + (half - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
        hi = np.flatnonzero(y[ipk+1:] < half)          # right crossing: first node after peak below half max
        if hi.size == 0:
            return np.nan                              # peak not resolved on the right
        j = ipk + 1 + hi[0]
        xhi = x[j-1] + (half - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1])
        return float(xhi - xlo)
    fwhm = np.empty(Ncompressed_tau)
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #Integration_____________________________________________________________________________________________________

    V0 = np.concatenate((a[0,:], b[0,:],f[0,:]))
    rec[0] = (*grid_energy(V0), 0.0, 0.0)
    fwhm[0] = _fwhm(zeta, np.abs(b[0,:])**2)
    P_in_prev, P_out_prev = edge_power(V0)
    pump_in_cum = 0.0; pump_out_cum = 0.0              # running time integrals of the edge powers (every step)
    next_save = 1
    for j in range(Ntau-1):
        N1 = N(V0)
        RS = BC(V0 + dtau * Tab_N[1,0]*N1)
        V1 = solve_banded((2,1), solve_M, RS)
        N2 = N(V1)
        LV1 = L@V1
        RS = BC(V0 + dtau * (Tab_L[2,1]*LV1 + Tab_N[2,0]*N1 + Tab_N[2,1]*N2))
        V2 = solve_banded((2,1), solve_M, RS)
        V0 = V2
        P_in, P_out = edge_power(V0)
        pump_in_cum  += 0.5*dtau*(P_in_prev + P_in)     # trapezoid in tau
        pump_out_cum += 0.5*dtau*(P_out_prev + P_out)
        P_in_prev, P_out_prev = P_in, P_out
        if next_save <Ncompressed_tau and j+1 == compressed_tau_indices[next_save]:
            a[next_save,:] = V0[:Nzeta]; b[next_save,:] = V0[Nzeta:2*Nzeta]; f[next_save,:] = V0[2*Nzeta:3*Nzeta]
            rec[next_save] = (*grid_energy(V0), pump_in_cum, pump_out_cum)
            fwhm[next_save] = _fwhm(zeta, np.abs(b[next_save,:])**2)
            next_save +=1
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #Energy diagnostics (one value per saved time)___________________________________________________________________
    pump_grid, seed_grid, ac_grid, pump_in, pump_out = rec.T
    reservoir = pump_in_cum - pump_in                  # pump not yet entered (vs end of run)

    total_energy = reservoir + pump_out + ac_grid + seed_grid + pump_grid
    E_in  = pump_grid[0] + pump_in                  # pump given up by the pump field
    E_out = (seed_grid - seed_grid[0]) + ac_grid + pump_out + pump_grid       # where it went
    eff   = (seed_grid - seed_grid[0]) / (pump_grid[0] + + pump_in)

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

    return a, b, f, fwhm, total_energy, (E_in, E_out), eff, phi_a, phi_b, phi_f


#example usage

sigmas       = [0.5,0.6]
b_0_initials = [1]
DETUNING = d    # detuning grid from the setup cell

runs = {}   # key: (b0, sigma) -> dict of arrays
for b0 in b_0_initials:
    for s in sigmas:
        a, b, f, fwhm, *_ = full_form(d=DETUNING, sigma=s, b0=b0)   # CHECK kwarg name
        runs[(b0, s)] = dict(a=a, b=b, f=f, fwhm=fwhm)
        print(f"done  b0 = {b0:<5}  sigma = {s}")

#plot
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
TIMES      = [130e-9]             # snapshot times (s) -- add as many as you like
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
t_idxs   = [np.argmin(np.abs(compressed_tau - T)) for T in TIMES]
t_actual = [compressed_tau[i] for i in t_idxs]
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

#alternatively, single plotting

a, b, f, fwhm, total_E, (E_in, E_out), eff, *_ = full_form(d = d)

#plotting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- toggles ----------------
SHOW_SEED   = True
SHOW_PUMP   = False
SHOW_AC     = False
SHOW_TARGET = True
SHOW_ANNOT  = True
SHOW_LEGEND = True
USE_XLIM    = True

# ---------------- settings ----------------
TIMES      = [400e-9, 600e-9,800e-9]               # snapshot times (s) -- add as many as you like
COLORS     = ["red", "green", "blue"]   # used in order; extra snapshots fall back to tab10
ANNOT_IDX  = -1                     # which snapshot the annotation reports (-1 = last)
XLIM       = (-2, 0.75)
FWHM_UNIT  = "m"
TARGET     = -1
PARAMS = {r"$g$": g, r"$\Gamma_B$": GammaB,
          r"$n$": n, r"$k_{norm}$": kNorm}

# ---------------- unit conversions ----------------
field_norm = (I0/2*1e-10) / 1000.0                   # |b|^2 -> MW/cm^2 -> GW/cm^2
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm    = 1000.0                                  # J/m^3 -> mJ/m^3

# ---------------- setup ----------------
colors   = [COLORS[k] if k < len(COLORS) else plt.cm.tab10(k % 10)
            for k in range(len(TIMES))]
t_idxs   = [np.argmin(np.abs(times_to_save - T)) for T in TIMES]
t_actual = [times_to_save[i] for i in t_idxs]
t_idx    = t_idxs[ANNOT_IDX]
pick     = lambda q: np.asarray(q)[t_idx] if np.ndim(q) else q
i_target = np.argmin(np.abs(zeta - TARGET))

# uses globals: a, b, f (shape (n_saves, Nzeta)), fwhm (n_saves,), zeta, times_to_save
fig, ax = plt.subplots(figsize=(7.5, 5.5))

I_snaps = [np.abs(b[ti, :])**2 * field_norm for ti in t_idxs]

# ---- seed ----
if SHOW_SEED:
    for I_k, col in zip(I_snaps, colors):
        ax.plot(zeta, I_k, color=col, linestyle="-")

if SHOW_TARGET:
    ax.axvline(TARGET, color="black", linestyle="--", alpha=0.5)

# ---- pump on its own axis ----
if SHOW_PUMP:
    ax_pump = ax.twinx()
    for ti, col in zip(t_idxs, colors):
        ax_pump.plot(zeta, np.abs(a[ti, :])**2 * field_norm,
                     color=col, linestyle="--", alpha=0.8)
    ax_pump.set_ylim(top=3*max(np.max(np.abs(a[ti, :])**2 * field_norm) for ti in t_idxs))
    ax_pump.set_ylabel(r"Pump (GW/cm$^2$)")

# ---- acoustic energy density on its own axis ----
if SHOW_AC:
    ax_ac = ax.twinx()
    if SHOW_PUMP:                          # keep clear of the pump axis
        ax_ac.spines['right'].set_position(('outward', 55))
    for ti, col in zip(t_idxs, colors):
        ax_ac.plot(zeta, np.abs(f[ti, :])**2 * u_ac * ac_norm,
                   color=col, linestyle=":", alpha=0.8)
    ax_ac.set_ylabel(r"Acoustic (mJ/m$^3$)")

# ---- annotation box ----
if SHOW_ANNOT:
    I_ann = I_snaps[ANNOT_IDX]
    ax.text(0.03, 0.95,
            f"peak = {I_ann.max():.3g} GW/cm$^2$\n"
            f"FWHM = {pick(fwhm):.3g} {FWHM_UNIT}\n"
            f"TARGET = {I_ann[i_target]:.3g} GW/cm$^2$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, zorder=5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

# snapshot-time key
for k, (t, col) in enumerate(zip(t_actual, colors)):
    ax.text(0.97, 0.95 - 0.07 * k, f"t = {t*1e9:.0f} ns",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=col, fontweight="bold", zorder=5)

ax.set_xlabel(r"$\zeta$ (m)")
if SHOW_SEED:
    ax.set_ylabel(r"Seed (GW/cm$^2$)")
else:
    ax.tick_params(labelleft=False)
ax.grid(alpha=0.3)

if USE_XLIM:
    ax.set_xlim(*XLIM)

if SHOW_LEGEND:
    handles = []
    if SHOW_SEED:
        handles.append(Line2D([0], [0], color="gray", linestyle="-",  label="Seed"))
    if SHOW_PUMP:
        handles.append(Line2D([0], [0], color="gray", linestyle="--", label="Pump"))
    if SHOW_AC:
        handles.append(Line2D([0], [0], color="gray", linestyle=":",  label="Acoustic"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)

title = ", ".join(f"{k} = {v:.7g}" for k, v in PARAMS.items())
if SHOW_ANNOT:
    title += f"   |   annotations at t = {t_actual[ANNOT_IDX]*1e9:.1f} ns"
ax.set_title(title, fontsize=11)
fig.tight_layout()
plt.show()

