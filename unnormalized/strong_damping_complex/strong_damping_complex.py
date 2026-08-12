#Computing and mounting the necessary constants------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

mu_0 = 4*np.pi*1e-7               ###                ###         #vacuum permeability
ep_0 = 8.854e-12          ###                    ###             #vacuum permittivity
c_light = 1/np.sqrt(mu_0 * ep_0) ###             ###             #m/s
rho_bar = 1.e-4        ###       ###          ###         ###    #n/a. small ad-hoc density perturbation scale.
n = 1.000495        ##        ##          ##            ##       #n/a. refractive index of Kr gas.
G_b = 4.132e7      ###             ###                          #Hz. Brillouin Linewidth of Kr gas
gamma = 3/2      ###                  ###                        #n/a. Adiabatic index
gamma_e = 2*(n-1)    ###         ###            ###              #n/a. Electrostriction coefficient
P = 101325         ###                          ###              #kg m-1 s-2. Gas pressure
rho_0 = 3.74    ###               ###                  ###       #kg/m3. Equilibrium gas density.
c_sound = np.sqrt(gamma*P/rho_0)      ###             ###        #m/s. Speed of sound in medium.
lambdaL = 248*1e-9        ###                ###                 #m. wavelength of laser light. 
f_laser = c_light/lambdaL      ###                   ###         #Hz. Frequency of laser light. 
w_laser = 2*np.pi*f_laser                ###               ###   #rad/s. Angular frequency of laser light. 
f_B = 1.627e9                   ###            ###              #Hz. Frequency of Brillouin Shift.
w_B = 2*np.pi*f_B                   ###            ###           #rad/s. Angular Frequency of Brillouin Shift.
K_B = w_B/c_sound      ###            ###                        #rad/m. Wavenumber of acoustic wave.
I_0 = 1.86e10    ###               ###               ###            #W/m2. Intensity of pump laser.
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
print(f"k = {kNorm}")
#Setup------------------------------------------------------------------------------------------
dtau = 0.05e-9
dzeta = 0.005
tau_range = [0, 100e-9]
zeta_range = [-10, 55]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta)

a0 = 1.5
b0 = 0.22
sigma = 1

#Integrator Function--------------------------------------------------------------------------------
def strong_damp_phase_dependent(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, g=g,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    Ka = (kNorm/2)*g
    Kb = (kNorm*c_light/n)*g

    A = np.empty((Ntau, Nzeta))
    B = np.empty((Ntau, Nzeta))
    phi_a = np.empty((Ntau, Nzeta))
    phi_b = np.empty((Ntau, Nzeta))

    A[0,:] = a0
    A[:,0] = a0
    B[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**2)   # seed profile

    for i in range(Ntau): #phi_a is constant in zeta. => phi_a is a function of i
        phi_a[i,:]  = np.pi*tau[i]/6
    for j in range(Nzeta): #
        phi_b[:,j] = np.pi*zeta[j]*1.5

    # pump pre-march at tau=0 (RK4 in zeta)
    for j in range(Nzeta - 1):
        aa = A[0,j]
        bb = B[0,j]
        db = B[0,j+1] - bb
        ka1 = -dzeta * Ka * aa * ( bb**2 )
        ka2 = -dzeta * Ka * (aa + ka1/2) * (bb + db/2)**2
        ka3 = -dzeta * Ka * (aa + ka2/2) * (bb + db/2)**2
        ka4 = -dzeta * Ka * (aa + ka3) * (bb + db)**2
        A[0,j+1] = aa + (ka1 + 2*ka2 + 2*ka3 + ka4)/6

    for i in range(Ntau - 1):
        # predictor: Euler in tau, RK4 in zeta
        db1 = dtau * Kb * B[i,:] * A[i,:]**2
        B[i+1,:] = B[i,:] + db1
        for j in range(Nzeta - 1):
            bb = B[i+1,j]
            db = B[i+1,j+1] - bb
            aa = A[i+1,j]
            da1 = -dzeta * Ka * aa * bb**2
            da2 = -dzeta * Ka * (aa + da1/2) * (bb+db/2)**2
            da3 = -dzeta * Ka * (aa + da2/2) * (bb + db/2)**2
            da4 = -dzeta * Ka * (aa + da3) * (bb + db)**2
            A[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
        # corrector: full RK4 in tau
        da = A[i+1,:] - A[i,:]
        db1 = dtau * Kb * B[i,:] * A[i,:]**2
        db2 = dtau * Kb * (B[i,:] + db1/2) * (A[i,:] + da/2)**2
        db3 = dtau * Kb * (B[i,:] + db2/2) * (A[i,:] + da/2)**2
        db4 = dtau * Kb * (B[i,:] + db3) * (A[i,:] + da)**2
        B[i+1,:] = B[i,:] + (db1 + 2*db2 + 2*db3 + db4)/6
        for j in range(Nzeta - 1):
            bb = B[i+1,j]
            db = B[i+1,j+1] - bb
            aa = A[i+1,j]
            da1 = -dzeta * Ka * aa * bb**2
            da2 = -dzeta * Ka * (aa + da1/2) * (bb+db/2)**2
            da3 = -dzeta * Ka * (aa + da2/2) * (bb + db/2)**2
            da4 = -dzeta * Ka * (aa + da3) * (bb + db)**2
            A[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6

    print(np.any(np.isinf(B)))
    print(np.any(np.isnan(B)))
    blown = np.where(np.isinf(B))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    a = A
    b = B
    f = g * A * B
    phi_f = phi_a - phi_b
    return a, b, f, phi_a, phi_b, phi_f
#Example usage------------------------------------------------------------------------------------------------
a, b, f, phi_a, phi_b, phi_f = strong_damp_phase_dependent()

#Plotting--------------------------------------------------------------------------------------------------------
t1 = 0
time1 = np.argmin(np.abs(tau-t1))
plt.plot(zeta, b[time1,:]**2, color="red", label=f"t = {t1 * 1e9}")
plt.plot(zeta, a[time1,:]**2, color="red", linestyle="--")
plt.plot(zeta, f[time1,:]**2, color="red", linestyle=":")

t2 = 43e-9
time2 = np.argmin(np.abs(tau-t2))
plt.plot(zeta, b[time2,:]**2, color="green", label=f"t = {t2 * 1e9}")
plt.plot(zeta, a[time2,:]**2, color="green", linestyle="--")
plt.plot(zeta, f[time2,:]**2, color="green", linestyle=":")

t3 = 80e-9
time3 = np.argmin(np.abs(tau-t3))
plt.plot(zeta, b[time3,:]**2, color="blue", label=f"t = {t3 * 1e9}")
plt.plot(zeta, a[time3,:]**2, color="blue", linestyle="--")
plt.plot(zeta, f[time3,:]**2, color="blue", linestyle=":")
plt.legend()
plt.xlabel("zeta (m)")
plt.ylabel("b^2")
#plt.xlim(left = -1.5, right = 0.5)
"""
