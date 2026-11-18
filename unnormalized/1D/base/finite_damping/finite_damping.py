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

#Longitudinal Pulse Parameters ##############################################
a0 = 1
b0 = 1
b_exp = 1
sigma = 1 #seed longitudinal gaussian width
phi_a_initial = 0 #initial pump phase
phi_b_initial = 0 #initial seed phase
#############################################################################

#integrator
# FINITE DAMPING case  -- 3wave structure: forward Euler on a, backward Euler on f (overdamped),
#                          RK4 on (a,f) when 2G<1, forward Euler on b.
#   da/dzeta = -(k/2) * b * f                 ,  Ka = k/2
#   db/dtau  =  (k*c_light/n) * a * conj(f)   ,  Kb = k*c_light/n
#   kappa*df/dzeta + f = g*a*conj(b)          ,  kappa = 2*c_light/(GammaB*n)
def finite_damp(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, b_exp=b_exp,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial,
                k=kNorm, g=g, n=n, c_light=c_light, GammaB=GammaB, G=G):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau, Nzeta = len(tau), len(zeta)
    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)
    f = np.empty((Ntau, Nzeta), dtype=complex)

    Ka    = k/2
    Kb    = k*c_light/n
    kappa = 2*c_light/(GammaB*n)

    # ICs: far-left for a and f, initial profile for b
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp) * np.exp(1j*phi_b_initial)
    a[:,0] = a0 * np.exp(1j*phi_a_initial)
    f[:,0] = 0

    # row-0 zeta march: forward Euler on a, backward Euler on f
    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] - dzeta * Ka * b[0,j] * f[0,j]
        f[0,j+1] = (g*a[0,j+1]*np.conj(b[0,j+1]) + (kappa/dzeta)*f[0,j]) / (1 + kappa/dzeta)

    # b kickstart: forward Euler   (db/dtau = Kb * a * conj(f))
    b[1,:] = b[0,:] + dtau * Kb * a[0,:] * np.conj(f[0,:])

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]; db = b[i+1,j+1] - bb
            aa = a[i+1,j]; ff = f[i+1,j]
            if 2*G > 1:
                a[i+1,j+1] = aa - dzeta * Ka * bb * ff
                f[i+1,j+1] = (g*a[i+1,j+1]*np.conj(bb+db) + (kappa/dzeta)*ff) / (1 + kappa/dzeta)
            else:
                da1 = -dzeta * Ka * bb * ff
                df1 =  dzeta * (1/kappa) * (g*aa*np.conj(bb) - ff)
                da2 = -dzeta * Ka * (bb + db/2) * (ff + df1/2)
                df2 =  dzeta * (1/kappa) * (g*(aa + da1/2)*np.conj(bb + db/2) - (ff + df1/2))
                da3 = -dzeta * Ka * (bb + db/2) * (ff + df2/2)
                df3 =  dzeta * (1/kappa) * (g*(aa + da2/2)*np.conj(bb + db/2) - (ff + df2/2))
                da4 = -dzeta * Ka * (bb + db) * (ff + df3)
                df4 =  dzeta * (1/kappa) * (g*(aa + da3)*np.conj(bb + db) - (ff + df3))
                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6
        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * a[i+1,:] * np.conj(f[i+1,:])

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    phi_f = np.angle(f)

    print(np.any(np.isinf(b)), np.any(np.isnan(b)))
    return a, b, f, phi_a, phi_b, phi_f

#example usage
a, b, f, phi_a, phi_b, phi_f = finite_damp()

#Plotting with Physical Units
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- unit conversions -------------------------------------------------
field_norm = (I0/2*1e-10) / 1000.0              # |a|^2,|b|^2 -> GW/cm^2
u_ac  = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm = 1.0 * 1000                            # J/m^3 -> mJ/m^3

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()                      # pump, same optical units
ax3 = ax1.twinx()                      # acoustic, its own units
ax3.spines['right'].set_position(('outward', 55))   # offset the 3rd axis

def plot_set(t_idx, color, label):
    ax1.plot(zeta, np.abs(b[t_idx, :])**2 * field_norm, color=color,
             linestyle="-",  label=label)
    ax2.plot(zeta, np.abs(a[t_idx, :])**2 * field_norm, color=color, linestyle="--")
    ax3.plot(zeta, np.abs(f[t_idx, :])**2 * u_ac * ac_norm,  color=color, linestyle=":")

t1 = 40e-9
time1 = np.argmin(np.abs(tau - t1))
plot_set(time1, "red",  f"t = {t1*1e9:.0f} ns")

t2 = 80e-9
time2 = np.argmin(np.abs(tau - t2))
plot_set(time2, "green",  f"t = {t2*1e9:.0f} ns")

t3 = 126.7e-9
time3 = np.argmin(np.abs(tau - t3))
plot_set(time3, "blue", f"t = {t3*1e9:.1f} ns")

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
left_min, left_max = ax1.get_ylim()
ax2.set_ylim(left_min * 0.2, left_max*0.2)
ax3.set_ylim(left_min *5, left_max*5)

plt.title("Finite Damping Amplification (Gaussian Seed Profile)")
plt.tight_layout()
ax1.grid()
plt.show()
