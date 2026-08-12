"""
Viable solver with pump ONLY focusing
pump has: radial gaussian, phase term
seed has: longitudinal gaussian
"""

#Setup of constants------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

G_b = 2.0974e9      ###             ###                          #Hz. Brillouin Linewidth of Kr gas
f_B = 1.5075e9                   ###            ###              #Hz. Frequency of Brillouin Shift.
rho_0 = 3.74    ###               ###                  ###       #kg/m3. Equilibrium gas density.
n = 1.000495        ##        ##          ##            ##       #n/a. refractive index of Kr gas.
gamma = 5/3      ###                  ###                        #n/a. Adiabatic index

mu_0 = 4*np.pi*1e-7               ###                ###         #vacuum permeability
ep_0 = 8.854e-12          ###                    ###             #vacuum permittivity
c_light = 1/np.sqrt(mu_0 * ep_0) ###             ###             #m/s
rho_bar = 1.e-4        ###       ###          ###         ###    #n/a. small ad-hoc density perturbation scale.
gamma_e = 2*(n-1)    ###         ###            ###              #n/a. Electrostriction coefficient
P = 101325         ###                          ###              #kg m-1 s-2. Gas pressure
c_sound = np.sqrt(gamma*P/rho_0)      ###             ###        #m/s. Speed of sound in medium.
lambdaL = 248*1e-9        ###                ###               #m. wavelength of laser light. 
f_laser = c_light/lambdaL      ###                   ###         #Hz. Frequency of laser light. 
w_laser = 2*np.pi*f_laser                ###               ###   #rad/s. Angular frequency of laser light. 
kL = w_laser/c_light                                             #rad/m. Wavenumber of laser light.
w_B = 2*np.pi*f_B                   ###            ###           #rad/s. Angular Frequency of Brillouin Shift.
K_B = w_B/c_sound      ###            ###                        #rad/m. Wavenumber of acoustic wave.
I_0 = 1e12    ###               ###               ###            #W/m2. Intensity of pump laser.
kNorm = rho_bar*gamma_e*w_laser/(4*c_light*n)                    #rad/m. normalization wavenumber.
wNorm = c_light * kNorm/n          ###             ###           #rad/s. Normalization frequency.
g = gamma_e * K_B * I_0 / (2 *G_b*c_light*c_sound*rho_0*rho_bar) #rad. gain coefficient
#I_0 = c_light*ep_0*E0**2
G = G_b / (2 * wNorm)                                            #1/rad. Normalized damping coefficient.
print(f"c_light = {c_light/1e8}*10^8m/s")
print(f"c_sound = {c_sound}m/s")
print(f"g = {g}")
print(f"G = {G}")
print(f"w_laser = {w_laser/1e15}*10^15Hz")
print(f"wNorm = {wNorm}Hz")
print(f"k = {kNorm}")

#Integration Setup ----------------------------------------------------------------------
dzeta = 0.01
drad = 1e-4
dtau =  0.8*dzeta / c_light

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

a0 = 1
b0 = 1
#sigma = 0.75
b_exp = 1
p_sigma = 2.8e-3

sigma_range = [0.75, 2.5]

#keep w0 on the order of a few mm OR increase rad_range to reduce the reflecting boundary effects
w0 = 2.75e-3
z_R = kL*w0**2 / 2
F = 80
times_to_save = np.arange(0,tau_range[1], 1e-9)
time_to_focus = (F / (1 + (F/z_R)**2)) *n/c_light
print("Time to Focus:", time_to_focus*1e9, "ns")
print("Distance to Focus:", time_to_focus*c_light/n, "m")
print(z_R)

#Integrator Function -----------------------------------------------------------------
def solver_2(times_to_save = times_to_save, b_exp = b_exp):
    """
practical 2D solver...
w/ pump focusing...
"""
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    rad  = np.arange(rad_range[0], rad_range[1], drad)

    Nzeta = len(zeta)
    Ntau  = len(tau)
    Nrad  = len(rad)

    kappa = 2*c_light/(n*G_b)

    ########################################################################################################################################
    #####TIMES TO SAVE -> GRID INDICES + OUTPUT BUFFERS#####################################################################################
    ########################################################################################################################################
    save_indices = sorted({int(np.argmin(np.abs(tau - t))) for t in times_to_save})
    pos = {idx: p for p, idx in enumerate(save_indices)}   # grid index -> row in *_out
    nsave = len(save_indices)
    saved_times = tau[save_indices]
    a_out = np.empty((nsave, Nzeta, Nrad), dtype=complex)
    b_out = np.empty((nsave, Nzeta, Nrad), dtype=complex)
    f_out = np.empty((nsave, Nzeta, Nrad), dtype=complex)

    # full-time accumulators (cheap; consumed by the 2D pocket_functs)
    b_axis    = np.empty((Ntau, Nzeta), dtype=float)   # |b| on axis r=0, all tau -> get_time_at
    a_in_col  = np.empty((Ntau, Nrad),  dtype=float)   # |a| at entrance column zeta[0]  -> get_eff
    a_out_col = np.empty((Ntau, Nrad),  dtype=float)   # |a| at exit column   zeta[-1]  -> get_eff

    ########################################################################################################################################
    #####MATRIX BANDS (identical to solver_1)###############################################################################################
    ########################################################################################################################################
    BK1 = np.zeros(Nrad, dtype=complex)
    BK2 = np.zeros(Nrad, dtype=complex)
    BK3 = np.zeros(Nrad, dtype=complex)
    for k in range(Nrad):
        if k == 0:  # 0 radius singularity
            BK1[k] = 0
            BK2[k] = c_light * dtau * (-1j)/(2*kL*n*drad**2)
            BK3[k] = c_light * dtau * (-1j)/(2*kL*n*drad**2)
        elif k == Nrad - 1:  # max radius condition
            BK1[k] = 0
            BK2[k] = 0 #turns into 1 in matrix
            BK3[k] = 0
        else:
            BK1[k] = (2*rad[k] - drad)*c_light*dtau*(-1j)/(16*n*rad[k]*kL*drad**2)
            BK2[k] = c_light*dtau*(-1j)/(4*n*kL*drad**2)
            BK3[k] = (2*rad[k] + drad)*c_light*dtau*(-1j)/(16*n*rad[k]*kL*drad**2)
    BM = np.zeros((3, Nrad), dtype=complex)
    BM[0, 1:]  = -BK3[:-1]    # upper diagonal — note the shift
    BM[1, :]   = 1 + BK2      # main  diagonal
    BM[2, :-1] = -BK1[1:]     # lower diagonal — note the shift

    AK1 = np.zeros(Nrad, dtype=complex)
    AK2 = np.zeros(Nrad, dtype=complex)
    AK3 = np.zeros(Nrad, dtype=complex)
    for k in range(Nrad):
        if k == 0:  # 0 radius singularity
            AK1[k] = 0
            AK2[k] = dzeta * (1j)/(4*kL*drad**2)
            AK3[k] = dzeta * (1j)/(4*kL*drad**2)
        elif k == Nrad - 1:  # max radius condition
            AK1[k] = 0
            AK2[k] = 0 #turns into 1 in matrix
            AK3[k] = 0
        else:
            AK1[k] = (2*rad[k] - drad)*dzeta*(1j)/(32*rad[k]*kL*drad**2)
            AK2[k] = dzeta*(1j)/(8*kL*drad**2)
            AK3[k] = (2*rad[k] + drad)*dzeta*(1j)/(32*rad[k]*kL*drad**2)
    AM = np.zeros((3, Nrad), dtype=complex)
    AM[0, 1:]  = -AK3[:-1]    # upper diagonal — note the shift
    AM[1, :]   = 1 + AK2      # main  diagonal
    AM[2, :-1] = -AK1[1:]     # lower diagonal — note the shift

    ########################################################################################################################################
    #####ADVANCE FUNCTIONS (slice-based; same math as solver_1)#############################################################################
    ########################################################################################################################################
    def advance_b(a_slice, f_slice, b_slice):
        """b[t] -> b[t+1]. Uses a[t], f[t] for the coupling. All (Nzeta, Nrad)."""
        A = np.zeros((Nrad, Nzeta), dtype=complex)
        for k in range(Nrad):
            if k == 0:
                A[k,:] = (1 - BK2[k])*b_slice[:,k] + BK3[k]*b_slice[:,k+1]
            elif k == Nrad - 1:
                A[k,:] = 0
            else:
                A[k,:] = BK1[k]*b_slice[:,k-1] + (1 - BK2[k])*b_slice[:,k] + BK3[k]*b_slice[:,k+1]
        bmid1 = solve_banded((1,1), BM, A).T
        bmid2 = bmid1 + dtau * (c_light*kNorm/n) * a_slice * np.conj(f_slice)

        for k in range(Nrad):
            if k == 0:
                A[k,:] = (1 - BK2[k])*bmid2[:,k] + BK3[k]*bmid2[:,k+1]
            elif k == Nrad - 1:
                A[k,:] = 0
            else:
                A[k,:] = BK1[k]*bmid2[:,k-1] + (1 - BK2[k])*bmid2[:,k] + BK3[k]*bmid2[:,k+1]
        return solve_banded((1,1), BM, A).T

    def advance_a_f_euler(a_slice, f_slice, b_slice, j):
        """Advance pump/phonon one zeta step: (a,f)[t, j] -> (a,f)[t, j+1]."""
        A = np.zeros(Nrad, dtype=complex)
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*a_slice[j,k-1] + (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
        amid1 = solve_banded((1,1), AM, A)
        amid2 = amid1 - dzeta * (kNorm/2)*b_slice[j,:]*f_slice[j,:]
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*amid2[k-1] + (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
        a_new = solve_banded((1,1), AM, A)
        f_new = (g*a_new*np.conj(b_slice[j+1,:]) + kappa/dzeta * f_slice[j,:]) / (1 + kappa/dzeta)
        return a_new, f_new

    def advance_a_f_RK4(a_slice, f_slice, b_slice, j):
        bb = b_slice[j,:]
        db = b_slice[j+1,:] - bb
        ff = f_slice[j,:]
        A = np.zeros(Nrad, dtype=complex)
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*a_slice[j,k-1] + (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
        amid1 = solve_banded((1,1), AM, A)

        da1 = -dzeta * kNorm/2 * bb*ff
        df1 = dzeta*(g*amid1*np.conj(bb) - ff)/kappa

        da2 = -dzeta * kNorm/2 * (bb+db/2)*(ff+df1/2)
        df2 = dzeta*(g*(amid1+da1/2)*np.conj(bb+db/2) - (ff+df1/2))/kappa

        da3 = -dzeta * kNorm/2 * (bb+db/2)*(ff+df2/2)
        df3 = dzeta*(g*(amid1+da2/2)*np.conj(bb+db/2) - (ff+df2/2))/kappa

        da4 = -dzeta * kNorm/2 * (bb+db)*(ff+df3)
        df4 = dzeta*(g*(amid1+da3)*np.conj(bb+db) - (ff+df3))/kappa
        amid2 = amid1 + (da1 + 2*da2 + 2*da3 + da4)/6
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*amid2[k-1] + (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
        a_new = solve_banded((1,1), AM, A)
        f_new = (g*a_new*np.conj(b_slice[j+1,:]) + kappa/dzeta * ff) / (1 + kappa/dzeta)
        return a_new, f_new

    def advance_a_t0(a_slice, j):
        """Undepleted pump at t=0: one zeta step of diffraction ONLY (no coupling).
        Mirrors solver_1.advance_a_t0_loop, slice-based (amid2 = amid1)."""
        A = np.zeros(Nrad, dtype=complex)
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*a_slice[j,k-1] + (1 - AK2[k])*a_slice[j,k] + AK3[k]*a_slice[j,k+1]
        amid1 = solve_banded((1,1), AM, A)
        amid2 = amid1
        for k in range(Nrad):
            if k == 0:
                A[k] = (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
            elif k == Nrad - 1:
                A[k] = 0
            else:
                A[k] = AK1[k]*amid2[k-1] + (1 - AK2[k])*amid2[k] + AK3[k]*amid2[k+1]
        a_new = solve_banded((1,1), AM, A)
        return a_new

    ########################################################################################################################################
    #####INITIAL CONDITIONS (time index 0)##################################################################################################
    ########################################################################################################################################
    # focused pump entrance profile at zeta[0], time-independent (a[:,0,:] in solver_1)
    pump_gaussian = np.exp(-(rad/w0)**2)
    a_entrance = a0 * pump_gaussian * np.exp(-1j*kL*rad**2/(2*F))   # (Nrad,)

    b_prev = np.empty((Nzeta, Nrad), dtype=complex)              # b[0]
    b_prev[:, :] = (b0 * np.exp(-np.abs(zeta/sigma)**b_exp))[:, None]
    f_slice = np.zeros((Nzeta, Nrad), dtype=complex)             # f[0] (no acoustic)

    # a[0]: undepleted pump, marched along zeta by diffraction only (solver_1 t=0 loop)
    a_slice = np.empty((Nzeta, Nrad), dtype=complex)
    a_slice[0, :] = a_entrance
    for j in range(Nzeta - 1):
        a_slice[j+1, :] = advance_a_t0(a_slice, j)

    if 0 in pos:  # save the t=0 snapshot (assigning into *_out copies the values)
        p = pos[0]
        a_out[p] = a_slice
        b_out[p] = b_prev
        f_out[p] = f_slice

    # t = 0 full-time accumulators
    b_axis[0, :]    = np.abs(b_prev[:, 0])
    a_in_col[0, :]  = np.abs(a_slice[0, :])
    a_out_col[0, :] = np.abs(a_slice[-1, :])

    b_curr = advance_b(a_slice, f_slice, b_prev)                 # b[1]

    ########################################################################################################################################
    #####LOOP###############################################################################################################################
    ########################################################################################################################################
    for i in range(Ntau - 1):
        t = i + 1
        # rebuild a[t], f[t] by marching zeta from the left BCs, using b[t] = b_curr
        a_slice = np.empty((Nzeta, Nrad), dtype=complex)
        f_slice = np.empty((Nzeta, Nrad), dtype=complex)
        a_slice[0, :] = a_entrance   # a[:,0,:] focused pump entrance (time-independent)
        f_slice[0, :] = 0            # f[:,0,:] = 0   (no acoustic at left)
        for j in range(Nzeta - 1):
            if 2*G > 1:  # stiff condition
                a_slice[j+1, :], f_slice[j+1, :] = advance_a_f_euler(a_slice, f_slice, b_curr, j)
            else:
                a_slice[j+1, :], f_slice[j+1, :] = advance_a_f_RK4(a_slice, f_slice, b_curr, j)

        # now a_slice = a[t], f_slice = f[t], b_curr = b[t]
        if t in pos:
            p = pos[t]
            a_out[p] = a_slice
            b_out[p] = b_curr
            f_out[p] = f_slice

        # full-time accumulators at time t (b_curr = b[t], a_slice = a[t])
        b_axis[t, :]    = np.abs(b_curr[:, 0])
        a_in_col[t, :]  = np.abs(a_slice[0, :])
        a_out_col[t, :] = np.abs(a_slice[-1, :])

        if i + 2 < Ntau:
            b_curr = advance_b(a_slice, f_slice, b_curr)         # b[t] -> b[t+1]
        if i % 100 == 0:
            print(f"Done tau = {round(tau[i]*1e9, 2)}ns")

    return np.abs(a_out), np.abs(b_out), np.abs(f_out), np.angle(a_out), np.angle(b_out), np.angle(f_out), b_axis, a_in_col, a_out_col

    
