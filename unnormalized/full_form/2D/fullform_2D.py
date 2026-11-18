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
dzeta = 0.01
dtau = 0.8 * dzeta/c_light
drad = 1e-4

zeta_range = [-2,2]
tau_range = [0,130e-9]
rad_range = [0,2e-2]

zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
tau = np.arange(tau_range[0], tau_range[1], dtau)
rad = np.arange(rad_range[0], rad_range[1], drad)

Nzeta = len(zeta)
Ntau = len(tau)
Nrad = len(rad)

t_true = tau
z_true = (zeta[None, :] - c_light*tau[:, None]/n)

#############################################################################

#compressed dimensions for saving###############################################################
compressed_tau = np.arange(0,tau_range[1], 1e-9)
compressed_tau_indices = sorted({int(np.argmin(np.abs(tau - t))) for t in compressed_tau})
Ncompressed_tau = len(compressed_tau_indices)

compressed_zeta = np.arange(zeta_range[0], zeta_range[1], 3*dzeta)
Ncompressed_zeta = len(compressed_zeta)
compressed_rad = np.arange(0, rad_range[1], 3*drad)
Ncompressed_rad = len(compressed_rad)
#############################################################################

#Longitudinal Pulse Parameters ##############################################
a0 = 1
b0 = 1
b_exp = 2
sigma = 1 #seed longitudinal gaussian width
d = np.zeros_like(zeta)
#############################################################################

#Focus and Radial Parameters#################################################

p_sigma = 5e-3 #pump radial supergaussian width
w0 = 2.45e-3 #w0<<r_max to reduce reflecting boundary effects
z_R = kL*w0**2 / 2
F = 80
time_to_focus = (F / (1 + (F/z_R)**2)) *n/c_light
print("Time to Focus:", time_to_focus*1e9, "ns")
print("Distance to Focus:", time_to_focus*c_light/n, "m")
print("Rayleigh range:", z_R, "m")
plt.figure(figsize=(4,3)); plt.plot(zeta, d); plt.xlabel("ζ (m)"); plt.ylabel("δ Detuning"); 
#############################################################################

#Tableau Parameters##########################################################
theta=1-1/np.sqrt(2)
eta = 1-1/(2*theta)

Tab_L = np.array([[0,0,0],[0,theta,0], [0,1-theta,theta]])
#Tab_L_weight = np.array([0,1-theta,theta])
Tab_N = np.array([[0,0,0],[theta,0,0],[eta,1-eta,0]])
#Tab_N_weight = np.array([eta,1-eta,0])

#integrator

def fullform_2D(d=d, diffraction=True, focusing=True, radial_pump = True, radial_seed = True, saving=False, savefile_suffix=""):
    """2D full-form
    Strang-Splitting:
    Crank-Nicolson Diffraction
    IMEX-RK ARS(2,2,2) field coupling
    Units: r [m], zeta [m], tau [s]. Energies [J], powers [W].
    Returns: a, b, f, fwhm, total_energy, (E_in, E_out), eff, phi_a, phi_b, phi_f   (same as the 1D full_form)
    Toggles:
        diffraction : if False, skip both strang_diffraction half steps
        focusing    : if False, no focusing phase on the initial seed
        saving      : if True, write everything to "<y|n>_diff_<y|n>_foc_<savefile_suffix>.npz"
                      (each entry is a dict: load with np.load(file, allow_pickle=True)["fields"].item())
    a, b, f (saved and returned) are on compressed_zeta x compressed_rad; diagnostics use the full grid.
    """
    #saved arrays live on the compressed grids; map each compressed point to its index on the full grid
    iz = np.rint((compressed_zeta - zeta[0])/dzeta).astype(int)
    ir = np.rint((compressed_rad - rad[0])/drad).astype(int)
    sub = np.ix_(iz, ir)                # full (Nzeta, Nrad) slice -> (len(compressed_zeta), len(compressed_rad))
    a = np.zeros((Ncompressed_tau, Ncompressed_zeta, Ncompressed_rad), dtype = complex)
    b = np.zeros((Ncompressed_tau, Ncompressed_zeta, Ncompressed_rad), dtype = complex)
    f = np.zeros((Ncompressed_tau, Ncompressed_zeta, Ncompressed_rad), dtype = complex)
    ########################################################################################################################################
    ####INITIAL CONDITIONS##################################################################################################################
    ########################################################################################################################################
    radial_seed_gaussian = np.exp(-(rad/w0)**2) if radial_seed else np.ones(Nrad)
    zeta_gaussian = np.exp(-np.abs(zeta/sigma)**b_exp)
    phase_term = np.exp(1j*kL*rad**2/(2*F)) if focusing else np.ones(Nrad)
    radial_pump_gaussian = np.exp(-(rad/p_sigma)**4) if radial_pump else np.ones(Nrad)

    a_curr = a0 * radial_pump_gaussian[None, :] * np.ones((Nzeta, 1), dtype=complex)                 #full-grid working arrays
    b_curr = b0 * zeta_gaussian[:, None] * radial_seed_gaussian[None, :] * phase_term[None, :]      #initial condition. Diffraction? phase?
    f_curr = np.zeros((Nzeta, Nrad), dtype=complex)
    a[0] = a_curr[sub]; b[0] = b_curr[sub]; f[0] = f_curr[sub]
    ########################################################################################################################################
    #####ENERGY DIAGNOSTICS + FWHM##########################################################################################################
    ########################################################################################################################################
    #Cross-section: annulus area of each radial cell (m^2). Axis cell = disk of radius drad/2, outer cell = half annulus.
    #Sum of weights = pi*rad[-1]^2 exactly. These weights also make the discrete radial Laplacian self-adjoint,
    #so the CN diffraction step conserves sum(w_r*|a|^2) exactly (PEC wall: no radial flux).
    w_r = 2*np.pi*rad*drad
    w_r[0] = np.pi*(drad/2)**2
    w_r[-1] = np.pi*(rad[-1]**2 - (rad[-1] - drad/2)**2)

    I_unit    = I0/2                                   # W/m^2 per |field|^2
    eps_light = (n/c_light) * I_unit                   # J/m^3 per |a|^2 or |b|^2
    eps_ac    = c_sound**2*rho_0*rho_bar**2/2          # J/m^3 per |f|^2
    v_pump    = 2*c_light/n                            # pump speed across the zeta boundaries
    def vol(F2):                                       # integral of F2 over zeta (trapezoid) and cross-section -> m^3
        return dzeta*((F2.sum(axis=0) - 0.5*(F2[0] + F2[-1])) @ w_r)
    def grid_energy(a_, b_, f_):                       # J on the grid: pump, seed, acoustic
        return (eps_light*vol(np.abs(a_)**2), eps_light*vol(np.abs(b_)**2), eps_ac*vol(np.abs(f_)**2))
    def edge_power(a_):                                # W: pump entering at zeta_0, leaving at zeta_f = (2c/n) * u * area
        return v_pump*eps_light*(np.abs(a_[0])**2 @ w_r), v_pump*eps_light*(np.abs(a_[-1])**2 @ w_r)
    rec = np.empty((Ncompressed_tau, 5))                   # per saved time: pump_grid, seed_grid, ac_grid, pump_in, pump_out

    #FWHM in zeta of the seed power profile P_b(zeta) = integral of |b|^2 over the cross-section
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
    seed_power = lambda b_: np.abs(b_[:,0])**2        # W-equivalent profile in zeta (|b|^2 * m^2)
    fwhm = np.empty(Ncompressed_tau)
    ########################################################################################################################################
    #####MATRIX BANDS#######################################################################################################################
    ########################################################################################################################################

    #First, the CN diffraction matrices (-AK1*a_rn-1   1+AK2*a_rn   -AK3*a_rn+1  = RS
    AK1 = np.zeros(Nrad, dtype=complex)
    AK1[0] = 0
    AK1[1:-1] = (2*rad[1:-1] - drad)*c_light*dtau*(1j)/(16*n*rad[1:-1]*kL*drad**2)
    AK1[-1] = 0                     # FIX: was nonzero, so the PEC row was not a pure identity row (gave a_rmax = AK1*a_rmax-1)

    AK2 = np.zeros(Nrad, dtype=complex)
    AK2[0] = c_light * dtau * (1j)/(2*kL*n*drad**2)
    AK2[1:-1] = c_light*dtau*(1j)/(4*n*kL*drad**2)
    AK2[-1] = 0

    AK3 = np.zeros(Nrad, dtype=complex)
    AK3[0] = c_light * dtau * (1j)/(2*kL*n*drad**2)
    AK3[1:-1] = (2*rad[1:-1] + drad)*c_light*dtau*(1j)/(16*n*rad[1:-1]*kL*drad**2)
    AK3[-1] = 0

    AM = np.zeros((3, Nrad), dtype=complex)
    AM[0, 1:]  = -AK3[:-1]    # upper diagonal — note the shift
    AM[1, :]   = 1 + AK2      # main  diagonal
    AM[2, :-1] = -AK1[1:]     # lower diagonal — note the shift

    BK1 = np.conj(AK1); BK2 = np.conj(AK2); BK3 = np.conj(AK3); BM = np.conj(AM)

    #IMEX coupling linear operator

    aL1 = np.ones(Nzeta) * 2*c_light/(n*2*dzeta);    aL1[0] = 0;   aL1[-1] = 2*c_light/(n*dzeta)
    aL2 = np.zeros(Nzeta);                           aL2[0] = 0;   aL2[-1] = -2*c_light/(n*dzeta)
    aL3 = np.ones(Nzeta) * -2*c_light/(n*2*dzeta);   aL3[0] = 0;   aL3[-1] = 0

    bL1 = np.zeros(Nzeta); bL2 = np.zeros(Nzeta); bL3 = np.zeros(Nzeta)

    fL1 = np.ones(Nzeta) *c_light/(n*2*dzeta);       fL1[0] = 0;   fL1[-1] = c_light/(n*dzeta)
    fL2 = np.ones(Nzeta) * -GammaB * (1-(1j)*d)/2;   fL2[0] = 0;   fL2[-1] = -c_light/(n*dzeta) - GammaB * (1-(1j)*d[-1])/2
    fL3 = np.ones(Nzeta) * -c_light/(n*2*dzeta);     fL3[0] = 0;   fL3[-1] = 0

    L1 = np.concatenate((aL1, bL1, fL1)); L2 = np.concatenate((aL2, bL2, fL2)); L3 = np.concatenate((aL3, bL3, fL3))
    LM = np.zeros((3, 3*Nzeta), dtype = complex)
    LM[0,1:] = L3[:-1]; LM[1,:] = L2; LM[2,:-1] = L1[1:]
    L = dia_array((LM, [1,0,-1]), shape=(3*Nzeta, 3*Nzeta))

    L_implicit = -dtau*theta*LM
    L_implicit[1, :] = L_implicit[1, :] + 1

    ########################################################################################################################################
    #####ADVANCE FUNCTIONS##################################################################################################################
    ########################################################################################################################################

    def N(V):
        N_da = -c_light * kNorm/n * V[Nzeta:2*Nzeta,:] * V[2*Nzeta:3*Nzeta,:]
        N_db = c_light * kNorm/n * V[:Nzeta,:] * np.conj(V[2*Nzeta:3*Nzeta,:])
        N_df = GammaB/2 * g * V[:Nzeta,:] * np.conj(V[Nzeta:2*Nzeta,:])
        return np.concatenate((N_da, N_db, N_df))

    def BC(V):
        V[0,:] = a0*radial_pump_gaussian[None, :]
        V[2*Nzeta,:] = 0            
        return V                    

    def strang_diffraction(a_curr, b_curr): #a_curr, b_curr given as arrays [zeta_i, rad_i]
        #operations run in parallel in different zeta. indexing reversal to set r to 1st index (temp).
        solve_A = np.zeros((Nrad, Nzeta), dtype = complex)
        solve_A[0,:] = AK1[0]*0 + (1-AK2[0])*a_curr[:,0] + AK3[0]*a_curr[:, 1]
        solve_A[1:-1,:] = AK1[1:-1,None]*a_curr[:,0:-2].T + (1-AK2[1:-1,None])*a_curr[:,1:-1].T + AK3[1:-1,None]*a_curr[:,2:].T   # FIX: [:,None]
        solve_A[-1,:] = 0

        solve_B = np.zeros((Nrad, Nzeta), dtype = complex) #Transpose needed from this. solve_banded only runs in parallel if operating over 1st dimension
        solve_B[0,:] = BK1[0]*0 + (1-BK2[0])*b_curr[:,0] + BK3[0]*b_curr[:, 1]
        solve_B[1:-1,:] = BK1[1:-1,None]*b_curr[:,0:-2].T + (1-BK2[1:-1,None])*b_curr[:,1:-1].T + BK3[1:-1,None]*b_curr[:,2:].T   # FIX: [:,None]
        solve_B[-1,:] = 0

        a_curr = solve_banded((1,1), AM, solve_A).T
        b_curr = solve_banded((1,1), BM, solve_B).T #transpose to bring zeta to 1st index.
        return a_curr, b_curr

    def strang_couple(a_curr, b_curr, f_curr):
        V0 = np.concatenate((a_curr, b_curr, f_curr))   # FIX: arrays must be passed as one tuple
        N1 = N(V0)
        RS = BC(V0 + dtau * Tab_N[1,0]*N1)
        V1 = solve_banded((1,1), L_implicit, RS) #no .T needed because matrix is solved in zeta, first dimension
        N2 = N(V1)
        LV1 = L@V1
        RS = BC(V0 + dtau * (Tab_L[2,1]*LV1 + Tab_N[2,0]*N1 + Tab_N[2,1]*N2))
        V2 = solve_banded((1,1), L_implicit, RS)
        a_curr = V2[:Nzeta,:]; b_curr = V2[Nzeta:2*Nzeta, :]; f_curr = V2[2*Nzeta:, :]
        return a_curr, b_curr, f_curr

    ########################################################################################################################################
    #####Integration Loops##################################################################################################################
    ########################################################################################################################################
    rec[0] = (*grid_energy(a_curr, b_curr, f_curr), 0.0, 0.0)
    fwhm[0] = _fwhm(zeta, seed_power(b_curr))
    P_in_prev, P_out_prev = edge_power(a_curr)
    pump_in_cum = 0.0; pump_out_cum = 0.0              # running time integrals of the edge powers (every step)
    next_save = 1                                      # FIX: was never initialised
    for j in range(Ntau-1):
        if diffraction:
            a_curr, b_curr = strang_diffraction(a_curr, b_curr)
        a_curr, b_curr, f_curr = strang_couple(a_curr, b_curr, f_curr)
        if diffraction:
            a_curr, b_curr = strang_diffraction(a_curr, b_curr)
        a_curr[0,:] = a0*radial_pump_gaussian          # inflow BC re-applied at the END of the step (was at the start),
                                                       # so saved states and edge powers see the BC profile
        P_in, P_out = edge_power(a_curr)
        pump_in_cum  += 0.5*dtau*(P_in_prev + P_in)     # trapezoid in tau
        pump_out_cum += 0.5*dtau*(P_out_prev + P_out)
        P_in_prev, P_out_prev = P_in, P_out

        if next_save <Ncompressed_tau and j+1 == compressed_tau_indices[next_save]:
            a[next_save] = a_curr[sub]; b[next_save] = b_curr[sub]; f[next_save] = f_curr[sub]   # compressed grids only
            rec[next_save] = (*grid_energy(a_curr, b_curr, f_curr), pump_in_cum, pump_out_cum)
            fwhm[next_save] = _fwhm(zeta, seed_power(b_curr))
            print(f"Done {round(compressed_tau[next_save]*1e9)}ns")
            next_save +=1 

    ########################################################################################################################################
    #####Energy diagnostics (one value per saved time)######################################################################################
    ########################################################################################################################################
    pump_grid, seed_grid, ac_grid, pump_in, pump_out = rec.T
    reservoir = pump_in_cum - pump_in                  # pump not yet entered (vs end of run)

    total_energy = reservoir + pump_out + ac_grid + seed_grid + pump_grid
    E_in  = pump_grid[0] + pump_in                     # pump given up by the pump field
    E_out = (seed_grid - seed_grid[0]) + ac_grid + pump_out + pump_grid       # where it went
    eff   = (seed_grid - seed_grid[0]) / (pump_grid[0] + pump_in)

    phi_a = np.angle(a); phi_b = np.angle(b); phi_f = np.angle(f)

    ########################################################################################################################################
    #####Save to npz########################################################################################################################
    ########################################################################################################################################
    if saving:
        yn = lambda flag: "y" if flag else "n"
        filename = f"{yn(diffraction)}_diff_{yn(focusing)}_foc" + (f"_{savefile_suffix}" if savefile_suffix else "") + ".npz"
        np.savez(filename,
                 gas_params         = {"GammaB": GammaB, "kNorm": kNorm, "g": g, "n": n, "kL": kL},
                 toggles            = {"diffraction": diffraction, "focusing": focusing, "radial pump": radial_pump, 
                                       "radial seed": radial_seed,"saving": saving},
                 dimension_grids    = {"compressed_zeta": compressed_zeta, "compressed_rad": compressed_rad,
                                       "compressed_tau": compressed_tau},
                 starting_params    = {"a0": a0, "b0": b0, "sigma": sigma, "b_exp": b_exp,
                                       "detuning": d, "z_R": z_R, "F": F},
                 fields             = {"a": a, "b": b, "f": f},
                 energy_diagnostics = {"fwhm": fwhm, "total_energy": total_energy, "E_in": E_in, "E_out": E_out, "eff": eff})
        print(f"saved {filename}")

    return a, b, f, fwhm, total_energy, (E_in, E_out), eff, phi_a, phi_b, phi_f

#usage
a, b, f,fwhm, E_tot, (Ein, Eout), eff, *_ = fullform_2D(diffraction = True, focusing=True, 
                                                        radial_pump = True, radial_seed = True,saving=True, savefile_suffix="pump_on")
#Single-Radius Plotting

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
TIMES      = [40e-9, 80e-9, 130e-9]     # snapshot times (s) -- add as many as you like
COLORS     = ["red", "green", "blue"]   # used in order; extra snapshots fall back to tab10
ANNOT_IDX  = -1                     # which snapshot the annotation reports (-1 = last)
XLIM       = (-2.5, 1.5)
FWHM_UNIT  = "m"
TARGET     = -1
RADS       = [0e-2, 0.5e-2, 1e-2]   # radii (m); one stacked panel per entry
PARAMS = {r"$g$": g, r"$\Gamma_B$": GammaB,
          r"$n$": n, r"$k_{norm}$": kNorm}

# ---------------- unit conversions ----------------
field_norm = (I0/2*1e-10) / 1000.0                   # |b|^2 -> MW/cm^2 -> GW/cm^2
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm    = 1000.0                                  # J/m^3 -> mJ/m^3

# ---------------- setup ----------------
colors   = [COLORS[k] if k < len(COLORS) else plt.cm.tab10(k % 10)
            for k in range(len(TIMES))]
t_idxs   = [np.argmin(np.abs(compressed_tau - T)) for T in TIMES]
t_actual = [compressed_tau[i] for i in t_idxs]
t_idx    = t_idxs[ANNOT_IDX]
pick     = lambda q: np.asarray(q)[t_idx] if np.ndim(q) else q
i_target = np.argmin(np.abs(compressed_zeta - TARGET))
r_idxs   = [np.argmin(np.abs(compressed_rad - R)) for R in RADS]

# uses globals: a, b, f (shape (n_saves, Nzeta, Nrad)), fwhm (n_saves,), compressed_*, 
nrow = len(RADS)
fig, axes = plt.subplots(nrow, 1, sharex=True, figsize=(7.5, 4.5*nrow), squeeze=False)
axes = axes[:, 0]

for row, (ax, R, ri) in enumerate(zip(axes, RADS, r_idxs)):
    I_snaps = [np.abs(b[ti, :, ri])**2 * field_norm for ti in t_idxs]

    # ---- seed ----
    if SHOW_SEED:
        for I_k, col in zip(I_snaps, colors):
            ax.plot(compressed_zeta, I_k, color=col, linestyle="-")

    if SHOW_TARGET:
        ax.axvline(TARGET, color="black", linestyle="--", alpha=0.5)

    # ---- pump on its own axis ----
    if SHOW_PUMP:
        ax_pump = ax.twinx()
        for ti, col in zip(t_idxs, colors):
            ax_pump.plot(compressed_zeta, np.abs(a[ti, :, ri])**2 * field_norm,
                         color=col, linestyle="--", alpha=0.8)
        ax_pump.set_ylim(top=3*max(np.max(np.abs(a[ti, :, ri])**2 * field_norm) for ti in t_idxs))
        ax_pump.set_ylabel(r"Pump (GW/cm$^2$)")

    # ---- acoustic energy density on its own axis ----
    if SHOW_AC:
        ax_ac = ax.twinx()
        if SHOW_PUMP:                          # keep clear of the pump axis
            ax_ac.spines['right'].set_position(('outward', 55))
        for ti, col in zip(t_idxs, colors):
            ax_ac.plot(compressed_zeta, np.abs(f[ti, :, ri])**2 * u_ac * ac_norm,
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

    if SHOW_SEED:
        ax.set_ylabel(r"Seed (GW/cm$^2$)")
    else:
        ax.tick_params(labelleft=False)
    ax.grid(alpha=0.3)

    if row == 0:                               # parameter title on the top panel only
        title = ", ".join(f"{k} = {v:.7g}" for k, v in PARAMS.items())
        if SHOW_ANNOT:
            title += f"   |   annotations at t = {t_actual[ANNOT_IDX]*1e9:.1f} ns"
        ax.set_title(title, fontsize=11)
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

    ax.text(0.03, 0.05, f"R = {R*1e2:.3g} cm", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold")

axes[-1].set_xlabel(r"$\zeta$ (m)")
if USE_XLIM:
    axes[0].set_xlim(*XLIM)                    # shared, applies to all

fig.suptitle("ζ Intensity Profile")
fig.tight_layout()
plt.show()

#cross-section propagation plotting
"""
Plots Intensity at each radius of select zeta against propagation distance
"""

field_norm = (I0/2*1e-10) / 1000.0                   # |b|^2 -> GW/cm^2 (same as before)

zi = np.argmin(np.abs(compressed_zeta - 0))
intensities = []
for ti in range(len(compressed_tau)):
    intensities.append(np.abs(b[ti, zi, :])**2)
intensities = np.array(intensities).T * field_norm   # was: / 10

def numeric_waist(b=b):
    zi = np.argmin(np.abs(compressed_zeta))
    waist_rad = []
    for ti in range(len(compressed_tau)):
        I = np.abs(b[ti][zi, :])**2                                  # intensity (b is |b|)
        den = np.trapezoid(I * compressed_rad, compressed_rad)
        num = np.trapezoid(I * compressed_rad**3, compressed_rad)
        waist_rad.append(np.sqrt(2 * num / den))               # sqrt(2*<r^2>)  <-- was the sqrt/threshold bug
    waist_rad = np.array(waist_rad)
    return waist_rad
numeric_waist = numeric_waist()


plt.figure()
dist = compressed_tau * c_light/n
plt.imshow(intensities, aspect="auto", origin = "lower", extent = [dist[0], dist[-1], compressed_rad[0], compressed_rad[-1]], cmap = "inferno")
plt.xlabel("Propagation Distance (m)")
plt.ylabel("Radius (m)")
plt.colorbar(label = r"Seed Intensity (GW/$cm^2$)") 
cs = plt.contour(dist, compressed_rad, intensities,
                 levels=6, colors='white', linewidths=0.6)
plt.title('Intensity at Seed Peak vs. Propagation Distance')
plt.plot(dist, numeric_waist, label="Numerical Radius", color="red")
plt.axvline((time_to_focus)*c_light/n, linestyle="--", color="white", label = "Analytical Waist Location (Pure Focusing)")
plt.ylim(top = 1e-2)
plt.legend()
plt.show()

#snapshot plotting
import numpy as np
import matplotlib.pyplot as plt

# Plots a full (zeta, r) snapshot of seed b, pump a, acoustic f at one time.
# Uses globals: a, b, f (shape (n_saves, Nzeta, Nrad)), compressed_tau, compressed_zeta,
#               compressed_rad, , I0, c_sound, rho_0, rho_bar, g, GammaB, n, kNorm

# ---------------- toggles ----------------
SHOW_SEED    = True
SHOW_PUMP    = True
SHOW_AC      = True
SHOW_TARGET  = True      # dashed lines at (target_ZETA, target_RAD) + value annotation at the intersection
USE_XLIM     = True
USE_YLIM     = True

# ---------------- settings ----------------
TIME   = 30e-9           # snapshot time (s)
XLIM   = (-2.5, 1.5)     # common to all panels
YLIM   = (0, 0.0075)     # common to all panels (m)
target_ZETA = -1         # m
target_RAD  = 0.5e-2     # m
CMAP   = "inferno"
PARAMS = {r"$g$": g, r"$\Gamma_B$": GammaB,
          r"$n$": n, r"$k_{norm}$": kNorm}

# ---------------- unit conversions ----------------
field_norm = (I0/2*1e-10) / 1000.0                   # |field|^2 -> GW/cm^2
u_ac       = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm    = 1000.0                                  # J/m^3 -> mJ/m^3

# ---------------- setup ----------------
ti       = np.argmin(np.abs(compressed_tau - TIME))
t_actual = compressed_tau[ti]
iz = np.argmin(np.abs(compressed_zeta - target_ZETA))
ir = np.argmin(np.abs(compressed_rad  - target_RAD))

panels = []   # (field, colorbar label, multiplier, annotation name, annotation unit)
if SHOW_SEED: panels.append((b, r"Seed Intensity (GW/cm$^2$)",  field_norm,     "Seed intensity",  r"GW/cm$^2$"))
if SHOW_PUMP: panels.append((a, r"Pump Intensity (GW/cm$^2$)",  field_norm,     "Pump intensity",  r"GW/cm$^2$"))
if SHOW_AC:   panels.append((f, r"Acoustic (mJ/m$^3$)",         u_ac * ac_norm, "Acoustic energy density", r"mJ/m$^3$"))

fig, axarr = plt.subplots(len(panels), 1, figsize=(20, 3.5*len(panels)),
                          sharex=True, sharey=True, squeeze=False)
axarr = axarr[:, 0]

for ax, (field, lab, m, name, unit) in zip(axarr, panels):
    I  = (np.abs(field[ti, :, :])**2).T * m          # shape (Nrad, Nzeta)
    im = ax.imshow(I, aspect="auto", origin="lower",
                   extent=[compressed_zeta[0], compressed_zeta[-1],
                           compressed_rad[0],  compressed_rad[-1]],
                   cmap=CMAP)
    fig.colorbar(im, ax=ax, label=lab)
    ax.set_ylabel("Radius (m)")
    if SHOW_TARGET:
        ax.axhline(target_RAD,  linestyle="--", color="white")
        ax.axvline(target_ZETA, linestyle="--", color="white")
        ax.plot(target_ZETA, target_RAD, "o", color="white", ms=4)
        ax.annotate(
                    f"{name} = {I[ir, iz]:.3g} {unit}\n"
                    rf"($\zeta$ = {compressed_zeta[iz]:.3g} m, r = {compressed_rad[ir]:.3g} m)",
                    xy=(0.02, 0.98), xycoords="axes fraction",  # <-- Fixed: top-left corner in axes fraction
                    xytext=(12, -12), textcoords="offset points",
                    ha="left", va="top", fontsize=8, zorder=5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

axarr[-1].set_xlabel(r"$\zeta$ (m)")
if USE_XLIM:
    axarr[0].set_xlim(*XLIM)     # shared, so applies to all
if USE_YLIM:
    axarr[0].set_ylim(*YLIM)

title = ", ".join(f"{k} = {v:.3g}" for k, v in PARAMS.items())
axarr[0].set_title(title, fontsize=11)
fig.suptitle(f"Intensity at Different Radii (t = {round(t_actual*1e9)} ns)")
fig.tight_layout()
plt.show()
