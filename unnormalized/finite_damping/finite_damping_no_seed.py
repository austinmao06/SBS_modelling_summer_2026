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
G = G_b / (2 * wNorm)                                            #1/rad. Normalized damping coefficient.
print(f"c_light = {c_light/1e8}*10^8m/s")
print(f"c_sound = {c_sound}m/s")
print(f"g = {g}")
print(f"G = {G}")
print(f"w_laser = {w_laser/1e15}*10^15Hz")
print(f"wNorm = {wNorm}Hz")
print(f"kNorm = {kNorm}")

#Setup------------------------------------------------------------------------------------------
dtau = 0.05e-9
dzeta = 0.005
tau_range = [0, 130e-9]
zeta_range = [-10, 5]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta)

a0 = 1.5
b0 = 0
sigma = 1
f0 = 0.22

#Integrator Function--------------------------------------------------------------------------------
def seedless(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
             a0=a0, f0=f0, sigma=sigma, g=g, G=G):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    Ka = kNorm/2
    Kb = kNorm*c_light/n
    kappa = 2*c_light/(G_b*n)

    a = np.empty((Ntau, Nzeta))
    b = np.empty((Ntau, Nzeta))
    f = np.empty((Ntau, Nzeta))

    b[0,:] = 0                                       # seedless: no initial seed
    a[:,0] = a0                                      # constant pump source
    f[:,0] = 0                                       # far-left limit of no acoustic
    f[0,:] = f0 * np.exp(-np.abs(zeta/sigma)**2)     # initial acoustic profile

    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] - dzeta * Ka * (b[0,j] * f[0,j])

    b[1,:] = b[0,:] + dtau * Kb * a[0,:] * f[0,:]

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]
            db = b[i+1,j+1] - bb
            aa = a[i+1,j]
            ff = f[i+1,j]

            if 2*G > 1:
                a[i+1, j+1] = aa - dzeta * Ka * bb * ff
                f[i+1,j+1] = (g * a[i+1,j+1] * (bb+db) + ff*kappa/dzeta )/(1 + kappa/dzeta)
            else:
                da1 = -dzeta * Ka * bb * ff
                df1 = dzeta / kappa * (g * aa * bb - ff)

                da2 = -dzeta * Ka * (bb + db/2) * (ff + df1/2)
                df2 = dzeta / kappa * (g * (aa + da1/2) * (bb + db/2) - (ff + df1/2))

                da3 = -dzeta * Ka * (bb + db/2) * (ff + df2/2)
                df3 = dzeta / kappa * (g * (aa + da2/2) * (bb + db/2) - (ff + df2/2))

                da4 = -dzeta * Ka * (bb + db) * (ff + df3)
                df4 = dzeta / kappa * (g * (aa + da3) * (bb + db) - (ff + df3))

                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6

        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * a[i+1,:] * f[i+1,:]

    print(np.any(np.isinf(b)))
    print(np.any(np.isnan(b)))
    blown = np.where(np.isinf(b))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    return a, b, f
#Example usage------------------------------------------------------------------------------------------------
a, b, f = seedless()

#Plotting--------------------------------------------------------------------------------------------------------
t1 = 0
time1 = np.argmin(np.abs(tau-t1))
plt.plot(zeta, b[time1,:]**2, color="red", label=f"t = {t1 * 1e9}ns")
plt.plot(zeta, a[time1,:]**2, color="red", linestyle="--")
plt.plot(zeta, f[time1,:]**2, color="red", linestyle=":")

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
