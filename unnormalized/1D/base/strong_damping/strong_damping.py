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
dtau = 0.025e-9                    # tau resolution (s). tau is TRUE time (s)

tau_range = [0, 50e-9]
zeta_range = [-30, 5]

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
sigma_normalized = 1                   #what is sigma in normalized version
sigma = sigma_normalized * (2/kNorm)   #seed longitudinal gaussian width, translated from the normalized frame
phi_a_initial = 0 #initial pump phase
phi_b_initial = 0 #initial seed phase

g = 1 #artificial g value to verify against normalized/2wave
#############################################################################

#integrator
# STRONG DAMPING case -- acoustic is slaved: f = g*a*conj(b)
#   da/dzeta = -(k/2) * g * a * |b|^2          ,  Ka = (k/2)*g
#   db/dtau  =  (k*c_light/n) * g * |a|^2 * b  ,  Kb = (k*c_light/n)*g
def strong_damp(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, b_exp=b_exp,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial,
                k=kNorm, g=g, n=n, c_light=c_light):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau, Nzeta = len(tau), len(zeta)
    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)

    Ka = (k/2) * g
    Kb = (k*c_light/n) * g

    # ICs: far-left / initial for a, initial profile for b
    a[0,:] = a0 * np.exp(1j*phi_a_initial)
    a[:,0] = a0 * np.exp(1j*phi_a_initial)
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp) * np.exp(1j*phi_b_initial)

    # row-0 zeta march: RK4 on a
    for j in range(Nzeta - 1):
        aa = a[0,j]; bb = b[0,j]; db = b[0,j+1] - bb
        ka1 = -dzeta * Ka * aa * np.abs(bb)**2
        ka2 = -dzeta * Ka * (aa + ka1/2) * np.abs(bb + db/2)**2
        ka3 = -dzeta * Ka * (aa + ka2/2) * np.abs(bb + db/2)**2
        ka4 = -dzeta * Ka * (aa + ka3) * np.abs(bb + db)**2
        a[0,j+1] = aa + (ka1 + 2*ka2 + 2*ka3 + ka4)/6

    # b kickstart: forward Euler   (db/dtau = Kb * |a|^2 * b)
    b[1,:] = b[0,:] + dtau * Kb * np.abs(a[0,:])**2 * b[0,:]

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            aa = a[i,j]; bb = b[i,j]; db = b[i,j+1] - bb
            ka1 = -dzeta * Ka * aa * np.abs(bb)**2
            ka2 = -dzeta * Ka * (aa + ka1/2) * np.abs(bb + db/2)**2
            ka3 = -dzeta * Ka * (aa + ka2/2) * np.abs(bb + db/2)**2
            ka4 = -dzeta * Ka * (aa + ka3) * np.abs(bb + db)**2
            a[i+1,j+1] = a[i+1,j] + (ka1 + 2*ka2 + 2*ka3 + ka4)/6
            # pump cannot overshoot through zero: a keeps the entrance phase, so clip when it flips sign along that phase
            if np.real(a[i+1,j+1] * np.conj(a[i+1,0])) < 0: a[i+1,j+1] = 0
        if i + 2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * np.abs(a[i+1,:])**2 * b[i+1,:]

    f = g * a * np.conj(b)   # slaved acoustic

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    phi_f = np.angle(f)

    print(np.any(np.isinf(b)), np.any(np.isnan(b)))
    return a, b, f, phi_a, phi_b, phi_f

#example usage
a, b, f, phi_a, phi_b, phi_f = strong_damp()

#plotting
t1 = 0
time1 = np.argmin(np.abs(tau-t1))
plt.plot(zeta, np.abs(b[time1,:])**2, color="red", label=f"t = {t1 * 1e9}ns")
plt.plot(zeta, np.abs(a[time1,:])**2, color="red", linestyle="--")
#plt.plot(zeta, np.abs(f[time1,:])**2, color="red", linestyle=":")

t2 = 10.64e-9
time2 = np.argmin(np.abs(tau-t2))
plt.plot(zeta, np.abs(b[time2,:])**2, color="green", label=f"t = {t2 * 1e9}ns")
plt.plot(zeta, np.abs(a[time2,:])**2, color="green", linestyle="--")
plt.plot(zeta, np.abs(f[time2,:])**2, color="green", linestyle=":")

t3 = 30e-9
time3 = np.argmin(np.abs(tau-t3))
plt.plot(zeta, np.abs(b[time3,:])**2, color="blue", label=f"t = {t3 * 1e9}ns")
plt.plot(zeta, np.abs(a[time3,:])**2, color="blue", linestyle="--")
plt.plot(zeta, np.abs(f[time3,:])**2, color="blue", linestyle=":")
plt.legend()
plt.xlabel(r"$\zeta$ (m)")
plt.ylabel(r"$|b|^2$")
plt.grid()
plt.show()
"""As you can see, t1, t2, t3 can be given in physical times now.
The numerical grid itself is built on physical units (seconds, meters).
This makes everything much easier to setup.
"""
