#Computing and mounting the necessary constants------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

mu_0 = 4*np.pi*1e-7               ###                ###         #vacuum permeability
ep_0 = 8.854e-12          ###                    ###             #vacuum permittivity
c_light = 1/np.sqrt(mu_0 * ep_0) ###             ###             #m/s
rho_bar = 1.e-4        ###       ###          ###         ###    #n/a. small ad-hoc density perturbation scale.
n = 1.000495        ##        ##          ##            ##       #n/a. refractive index of Kr gas.
G_b = 2.09741e9      ###             ###                          #Hz. Brillouin Linewidth of Kr gas
gamma = 5/3      ###                  ###                        #n/a. Adiabatic index
gamma_e = 2*(n-1)    ###         ###            ###              #n/a. Electrostriction coefficient
P = 101325         ###                          ###              #kg m-1 s-2. Gas pressure
rho_0 = 3.74    ###               ###                  ###       #kg/m3. Equilibrium gas density.
c_sound = np.sqrt(gamma*P/rho_0)      ###             ###        #m/s. Speed of sound in medium.
lambdaL = 248*1e-9        ###                ###                 #m. wavelength of laser light. 
f_laser = c_light/lambdaL      ###                   ###         #Hz. Frequency of laser light. 
w_laser = 2*np.pi*f_laser                ###               ###   #rad/s. Angular frequency of laser light. 
f_B = 1.5075e9                   ###            ###              #Hz. Frequency of Brillouin Shift.
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
print(f"kNorm = {kNorm}")

#Setup------------------------------------------------------------------------------------------

dtau = 0.025e-9
dzeta = 0.01
tau_range = [0, 50e-9]
zeta_range = [-30, 5]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta)

a0 = 1
b0 = 1
sigma_normalized = 1 #what is sigma in normalized version
sigma =  sigma_normalized * (2/kNorm)   # sigma = 0.091 in normalized frame translated to 

g = 1 #artificial g value to verify against normalized/2wave
#Integrator Function--------------------------------------------------------------------------------
def strong_damp(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, b_exp=1,
                k=kNorm, g=g, n=n, c_light=c_light):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau, Nzeta = len(tau), len(zeta)
    a = np.empty((Ntau, Nzeta))
    b = np.empty((Ntau, Nzeta))

    Ka = (k/2) * g
    Kb = (k*c_light/n) * g

    # ICs: far-left / initial for a, initial profile for b
    a[0,:] = a0
    a[:,0] = a0
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp)

    # row-0 zeta march: RK4 on a
    for j in range(Nzeta - 1):
        aa = a[0,j]; bb = b[0,j]; db = b[0,j+1] - bb
        ka1 = -dzeta * Ka * aa * (bb**2)
        ka2 = -dzeta * Ka * (aa + ka1/2) * (bb + db/2)**2
        ka3 = -dzeta * Ka * (aa + ka2/2) * (bb + db/2)**2
        ka4 = -dzeta * Ka * (aa + ka3) * (bb + db)**2
        a[0,j+1] = aa + (ka1 + 2*ka2 + 2*ka3 + ka4)/6

    # b kickstart: forward Euler   (db/dtau = Kb * a^2 * b)
    b[1,:] = b[0,:] + dtau * Kb * (a[0,:]**2) * b[0,:]

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            aa = a[i,j]; bb = b[i,j]; db = b[i,j+1] - bb
            ka1 = -dzeta * Ka * aa * bb**2
            ka2 = -dzeta * Ka * (aa + ka1/2) * (bb + db/2)**2
            ka3 = -dzeta * Ka * (aa + ka2/2) * (bb + db/2)**2
            ka4 = -dzeta * Ka * (aa + ka3) * (bb + db)**2
            a[i+1,j+1] = a[i+1,j] + (ka1 + 2*ka2 + 2*ka3 + ka4)/6
            if a[i+1,j+1] < 0: a[i+1,j+1] = 0
        if i + 2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * (a[i+1,:]**2) * b[i+1,:]

    f = g * a * b   # slaved acoustic
    print(np.any(np.isinf(b)), np.any(np.isnan(b)))
    return a, b, f
#Example usage------------------------------------------------------------------------------------------------
a, b, f = strong_damp()

#Plotting--------------------------------------------------------------------------------------------------------
t1 = 0
time1 = np.argmin(np.abs(tau-t1))
plt.plot(zeta, b[time1,:]**2, color="red", label=f"t = {t1 * 1e9}ns")
plt.plot(zeta, a[time1,:]**2, color="red", linestyle="--")
#plt.plot(zeta, f[time1,:]**2, color="red", linestyle=":")

t2 = 10.64e-9 
time2 = np.argmin(np.abs(tau-t2))
plt.plot(zeta, b[time2,:]**2, color="green", label=f"t = {t2 * 1e9}ns")
plt.plot(zeta, a[time2,:]**2, color="green", linestyle="--")
plt.plot(zeta, f[time2,:]**2, color="green", linestyle=":")

t3 = 30e-9
time3 = np.argmin(np.abs(tau-t3))
plt.plot(zeta, b[time3,:]**2, color="blue", label=f"t = {t3 * 1e9}ns")
plt.plot(zeta, a[time3,:]**2, color="blue", linestyle="--")
plt.plot(zeta, f[time3,:]**2, color="blue", linestyle=":")
plt.legend()
plt.xlabel(r"$\zeta$ (m)")
plt.ylabel("b^2")
plt.grid()
"""As you can see, t1, t2, t3 can be given in physical times now.
The numerical grid itself is built on physical units (seconds, meters).
This makes everything much easier to setup.
"""
