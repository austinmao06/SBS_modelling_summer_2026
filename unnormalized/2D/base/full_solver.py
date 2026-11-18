"""
No detuning. This is the full solver as described by the numerical methods: every tau step of a, b, f is stored.
It is NOT FOR USING. TOO COMPUTATIONALLY EXPENSIVE (memory ~ Ntau*Nzeta*Nrad per field).
Use viable_solver.py for identical results. Same toggles as viable_solver.
"""


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

zeta_range = [-25,5]
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

#Longitudinal Pulse Parameters ##############################################
a0 = 1
b0 = 1
b_exp = 1
sigma = 0.75 #seed longitudinal gaussian width
sigma_range = [0.75, 2.5] #seed widths spread across the radial columns (sigma_sweep toggle only)
#############################################################################

#Focus and Radial Parameters#################################################

p_sigma = 2.8e-3 #pump radial supergaussian width
w0 = 2.75e-3 #w0<<r_max to reduce reflecting boundary effects
z_R = kL*w0**2 / 2
F = 80
time_to_focus = (F / (1 + (F/z_R)**2)) *n/c_light
print("Time to Focus:", time_to_focus*1e9, "ns")
print("Distance to Focus:", time_to_focus*c_light/n, "m")
print("Rayleigh range:", z_R, "m")
#############################################################################

#Pump Shutoff Parameters (pump_off toggle only)##############################
time_to_off = 127e-9       #lab time (s) at which the entrance pump begins to drop
decrease_duration = 3e-9   #time (s) over which it falls from full to 0
#############################################################################

#integrator

def full_solver(diffraction=True, focusing=True, pump_focusing=False, radial_pump=True, radial_seed=True,
                sigma_sweep=False, pump_off=False, time_to_off=time_to_off, decrease_duration=decrease_duration,
                saving=False, savefile_suffix=""):
    """2D solver full scheme (NOT FOR USE)...
    cylindrical coordinates, radial symmetry...
    Strang-Splitting:
    Crank-Nicolson + F/B-Euler or RK4
    Returns: a, b, f, phi_a, phi_b, phi_f   (complex fields at every tau, shape (Ntau, Nzeta, Nrad))
    Toggles: identical to viable_solver (see its docstring). saving writes
        "full_<y|n>_diff_<y|n>_foc[_pfoc][_pumpoff][_sweep]_<savefile_suffix>.npz"
    """
    kappa = 2*c_light/(n*GammaB)
    if sigma_sweep and diffraction:
        print("WARNING: sigma_sweep with diffraction=True couples the radial columns; they are no longer independent 1D runs")

    a = np.zeros((Ntau, Nzeta, Nrad), dtype = complex)
    b = np.zeros((Ntau, Nzeta, Nrad), dtype = complex)
    f = np.zeros((Ntau, Nzeta, Nrad), dtype = complex)
    ########################################################################################################################################
    #####MATRIX BANDS#######################################################################################################################
    ########################################################################################################################################
    #seed: CN half step in tau.  -BK1*b_k-1 + (1+BK2)*b_k - BK3*b_k+1 = BK1*b_k-1 + (1-BK2)*b_k + BK3*b_k+1
    BK1 = np.zeros(Nrad, dtype=complex); BK2 = np.zeros(Nrad, dtype=complex); BK3 = np.zeros(Nrad, dtype=complex)
    BK2[0] = c_light * dtau * (-1j)/(2*kL*n*drad**2)                  # r = 0 singularity
    BK3[0] = c_light * dtau * (-1j)/(2*kL*n*drad**2)
    BK1[1:-1] = (2*rad[1:-1] - drad)*c_light*dtau*(-1j)/(16*n*rad[1:-1]*kL*drad**2)
    BK2[1:-1] = c_light*dtau*(-1j)/(4*n*kL*drad**2)
    BK3[1:-1] = (2*rad[1:-1] + drad)*c_light*dtau*(-1j)/(16*n*rad[1:-1]*kL*drad**2)
    #BK*[-1] = 0: max radius (PEC) row is the identity
    BM = np.zeros((3, Nrad), dtype=complex)
    BM[0, 1:]  = -BK3[:-1]    # upper diagonal — note the shift
    BM[1, :]   = 1 + BK2      # main  diagonal
    BM[2, :-1] = -BK1[1:]     # lower diagonal — note the shift

    #pump: CN half step in zeta
    AK1 = np.zeros(Nrad, dtype=complex); AK2 = np.zeros(Nrad, dtype=complex); AK3 = np.zeros(Nrad, dtype=complex)
    AK2[0] = dzeta * (1j)/(4*kL*drad**2)                               # r = 0 singularity
    AK3[0] = dzeta * (1j)/(4*kL*drad**2)
    AK1[1:-1] = (2*rad[1:-1] - drad)*dzeta*(1j)/(32*rad[1:-1]*kL*drad**2)
    AK2[1:-1] = dzeta*(1j)/(8*kL*drad**2)
    AK3[1:-1] = (2*rad[1:-1] + drad)*dzeta*(1j)/(32*rad[1:-1]*kL*drad**2)
    AM = np.zeros((3, Nrad), dtype=complex)
    AM[0, 1:]  = -AK3[:-1]    # upper diagonal — note the shift
    AM[1, :]   = 1 + AK2      # main  diagonal
    AM[2, :-1] = -AK1[1:]     # lower diagonal — note the shift

    ########################################################################################################################################
    #####ADVANCE FUNCTIONS##################################################################################################################
    ########################################################################################################################################
    def cn_rhs(K1, K2, K3, x):
        """explicit side of a CN half step along the radial (last) axis; PEC row at r_max -> 0"""
        R = np.zeros_like(x, dtype=complex)
        R[..., 0]    = (1 - K2[0])*x[..., 0] + K3[0]*x[..., 1]
        R[..., 1:-1] = K1[1:-1]*x[..., :-2] + (1 - K2[1:-1])*x[..., 1:-1] + K3[1:-1]*x[..., 2:]
        return R

    def diffract_b(b_slice):          # (Nzeta, Nrad). solve_banded only runs in parallel over the 1st dimension -> transpose
        return solve_banded((1,1), BM, cn_rhs(BK1, BK2, BK3, b_slice).T).T

    def diffract_a(a_row):            # (Nrad,)
        return solve_banded((1,1), AM, cn_rhs(AK1, AK2, AK3, a_row))

    def advance_b_euler(i_idx_in):
        """b[i] -> b[i+1]: half CN diffraction step. Full Forward Euler step. half CN diffraction step"""
        bmid1 = diffract_b(b[i_idx_in]) if diffraction else b[i_idx_in]
        bmid2 = bmid1 + dtau * (c_light*kNorm/n) * a[i_idx_in] * np.conj(f[i_idx_in])
        return diffract_b(bmid2) if diffraction else bmid2

    def advance_a_f_euler(i_idx_in, j_idx_in):
        amid1 = diffract_a(a[i_idx_in, j_idx_in, :]) if diffraction else a[i_idx_in, j_idx_in, :]
        amid2 = amid1 - dzeta * (kNorm/2)*b[i_idx_in, j_idx_in, :]*f[i_idx_in, j_idx_in, :]
        a_new = diffract_a(amid2) if diffraction else amid2
        f_new = (g*a_new*np.conj(b[i_idx_in, j_idx_in+1, :]) + kappa/dzeta * f[i_idx_in, j_idx_in, :]) / (1 + kappa/dzeta)
        return a_new, f_new

    def advance_a_f_RK4(i_idx_in, j_idx_in):
        bb = b[i_idx_in, j_idx_in, :]
        db = b[i_idx_in, j_idx_in+1, :] - bb
        ff = f[i_idx_in, j_idx_in, :]
        amid1 = diffract_a(a[i_idx_in, j_idx_in, :]) if diffraction else a[i_idx_in, j_idx_in, :]

        da1 = -dzeta * kNorm/2 * bb*ff
        df1 = dzeta*(g*amid1*np.conj(bb) - ff)/kappa

        da2 = -dzeta * kNorm/2 * (bb+db/2)*(ff+df1/2)
        df2 = dzeta*(g*(amid1+da1/2)*np.conj(bb+db/2) - (ff+df1/2))/kappa

        da3 = -dzeta * kNorm/2 * (bb+db/2)*(ff+df2/2)
        df3 = dzeta*(g*(amid1+da2/2)*np.conj(bb+db/2) - (ff+df2/2))/kappa

        da4 = -dzeta * kNorm/2 * (bb+db)*(ff+df3)
        df4 = dzeta*(g*(amid1+da3)*np.conj(bb+db) - (ff+df3))/kappa
        amid2 = amid1 + (da1 + 2*da2 + 2*da3 + da4)/6
        a_new = diffract_a(amid2) if diffraction else amid2
        f_new = (g*a_new*np.conj(b[i_idx_in, j_idx_in+1, :]) + kappa/dzeta * ff) / (1 + kappa/dzeta)
        return a_new, f_new

    def advance_a_t0(a_row):
        """Undepleted pump at t=0: one zeta step of diffraction ONLY (no coupling)."""
        return diffract_a(diffract_a(a_row))

    ########################################################################################################################################
    ####INITIAL CONDITIONS##################################################################################################################
    ########################################################################################################################################
    radial_seed_gaussian = np.exp(-(rad/w0)**2) if radial_seed else np.ones(Nrad)
    zeta_gaussian = np.exp(-np.abs(zeta/sigma)**b_exp)
    phase_term = np.exp(1j*kL*rad**2/(2*F)) if focusing else np.ones(Nrad)
    if pump_focusing:
        pump_prof = a0 * np.exp(-(rad/w0)**2) * np.exp(-1j*kL*rad**2/(2*F))   # focused gaussian pump
    elif radial_pump:
        pump_prof = a0 * np.exp(-(rad/p_sigma)**4)                            # radial supergaussian pump
    else:
        pump_prof = a0 * np.ones(Nrad)                                        # flat pump
    pump_prof = pump_prof.astype(complex)

    #timed pump gate at the entrance (scalar per time step)
    pump_gate = np.ones(Ntau)
    if pump_off:
        sg_sigma  = decrease_duration / 3                                   # 3 sigma = full decrease_duration
        pump_gate = np.where(tau <= time_to_off, 1.0,
                             np.exp(-((tau - time_to_off)/sg_sigma)**4))     # 1 -> 4th-power super-gaussian
        pump_gate[tau >= time_to_off + decrease_duration] = 0.0             # clip tail to exactly 0

    if sigma_sweep:
        sigmas_r = np.linspace(sigma_range[0], sigma_range[1], Nrad)
        b[0,:,:] = b0 * np.exp(-np.abs(zeta[:, None]/sigmas_r[None, :])**b_exp)
    else:
        b[0,:,:] = b0 * zeta_gaussian[:, None] * radial_seed_gaussian[None, :] * phase_term[None, :]

    a[:,0,:] = pump_prof[None, :] * pump_gate[:, None]     # pump entrance at zeta[0], every tau
    for j in range(Nzeta - 1):                             # a[0]: undepleted pump
        a[0,j+1,:] = advance_a_t0(a[0,j,:]) if diffraction else a[0,j,:]
    #f[:,0,:] = 0 and f[0,:,:] = 0 (no acoustic) from np.zeros

    ########################################################################################################################################
    #####LOOP###############################################################################################################################
    ########################################################################################################################################
    b[1, :, :] = advance_b_euler(0)

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            if 2*G > 1: #stiff condition
                a[i+1, j+1,:], f[i+1,j+1,:] = advance_a_f_euler(i+1,j)
            else:
                a[i+1, j+1,:], f[i+1,j+1,:] = advance_a_f_RK4(i+1,j)

        if i + 2 < Ntau:
            b[i+2,:,:] = advance_b_euler(i+1)
        if i % 100 == 0:
            print(f"Done tau = {tau[i]*1e9}ns")

    phi_a = np.angle(a); phi_b = np.angle(b); phi_f = np.angle(f)

    ########################################################################################################################################
    #####Save to npz########################################################################################################################
    ########################################################################################################################################
    if saving:
        yn = lambda flag: "y" if flag else "n"
        filename = (f"full_{yn(diffraction)}_diff_{yn(focusing)}_foc" + ("_pfoc" if pump_focusing else "")
                    + ("_pumpoff" if pump_off else "") + ("_sweep" if sigma_sweep else "")
                    + (f"_{savefile_suffix}" if savefile_suffix else "") + ".npz")
        np.savez(filename,
                 physical_params = {"GammaB": GammaB, "OmegaB": OmegaB, "rho_0": rho_0, "n": n, "gamma": gamma, "dlambda": 0},
                 run_params = {"a0": a0, "p_sigma": p_sigma, "b0": b0, "sigma": sigma, "b_exp": b_exp, "w0": w0, "F": F,
                               "focus_dist": time_to_focus*c_light/n, "sigma_range": sigma_range,
                               "time_to_off": time_to_off, "decrease_duration": decrease_duration},
                 toggles = {"diffraction": diffraction, "focusing": focusing, "pump_focusing": pump_focusing,
                            "radial_pump": radial_pump, "radial_seed": radial_seed, "sigma_sweep": sigma_sweep,
                            "pump_off": pump_off},
                 a=a, b=b, f=f,
                 rad=rad, zeta=zeta, tau=tau,
                 phi_a = phi_a, phi_b = phi_b, phi_f = phi_f, b_axis = np.abs(b[:, :, 0]),
                 a_in_col = np.abs(a[:, 0, :]), a_out_col = np.abs(a[:, -1, :]))
        print(f"saved {filename}")

    return a, b, f, phi_a, phi_b, phi_f

#usage (commented out: the full grid needs tens of GB of memory -- shrink the grids in the setup cell first)
"""
a, b, f, phi_a, phi_b, phi_f = full_solver(diffraction=True, focusing=True, pump_focusing=False,
                                           radial_pump=True, radial_seed=True, sigma_sweep=False,
                                           pump_off=False, saving=True, savefile_suffix="")
"""
