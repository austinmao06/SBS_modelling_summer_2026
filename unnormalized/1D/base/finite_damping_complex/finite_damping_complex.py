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

phi_a_initial = 0
phi_b_initial = 0

#Integrator Function--------------------------------------------------------------------------------
"""the below function explicitly uses real and complex to match the OVERVIEW equations"""
def finite_damp_phase_dependent(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, g=g, G=G,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    Ka = kNorm/2
    Kb = kNorm*c_light/n
    kappa = 2*c_light/(G_b*n)
    
    aR = np.empty((Ntau, Nzeta))
    aI = np.empty((Ntau, Nzeta))
    bR = np.empty((Ntau, Nzeta))
    bI = np.empty((Ntau, Nzeta))
    fR = np.empty((Ntau, Nzeta))
    fI = np.empty((Ntau, Nzeta))
    
    phi_a = np.empty((Ntau, Nzeta))
    phi_b = np.empty((Ntau, Nzeta))
    phi_f = np.empty((Ntau, Nzeta))
    ###PARAMS########################
    
    b0_profile = b0 * np.exp(-np.abs(zeta/sigma)**2)
    
    aR[:,0] = a0 * np.cos(phi_a_initial)
    aI[:,0] = a0 * np.sin(phi_a_initial)
    
    bR[0,:] = b0_profile * np.cos(phi_b_initial)
    bI[0,:] = b0_profile * np.sin(phi_b_initial)
    
    fR[:,0] = 0   # far-left column: no incoming acoustic (keep)
    fI[:,0] = 0
    
    for j in range(Nzeta - 1):
        aR[0,j+1] = aR[0,j] + dzeta * Ka * (bI[0,j] * fI[0,j] - bR[0,j] * fR[0,j])
        aI[0,j+1] = aI[0,j] + dzeta * Ka * (-bR[0,j] * fI[0,j] - bI[0,j] * fR[0,j])
        fR[0,j+1] = (g * aR[0,j+1] * bR[0,j+1] + g * aI[0,j+1] * bI[0,j+1] + fR[0,j]*kappa/dzeta)/(1 + kappa/dzeta)
        fI[0,j+1] = (g * aI[0,j+1] * bR[0,j+1] - g * aR[0,j+1] * bI[0,j+1] + fI[0,j]*kappa/dzeta)/(1 + kappa/dzeta)
    
    bR[1,:] = bR[0,:] + dtau * Kb * (aR[0,:] * fR[0,:] + aI[0,:] * fI[0,:])
    bI[1,:] = bI[0,:] + dtau * Kb * (-aR[0,:] * fI[0,:] + aI[0,:] * fR[0,:])
    
    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bbR = bR[i+1,j]
            dbR = bR[i+1,j+1] - bbR
            bbI = bI[i+1,j]
            dbI = bI[i+1,j+1] - bbI
            aaR = aR[i+1,j]
            aaI = aI[i+1,j]
            ffR = fR[i+1,j]
            ffI = fI[i+1,j]
    
            if 2*G > 1:
                aR[i+1,j+1] = aaR + dzeta * Ka * (bbI * ffI - bbR * ffR)
                aI[i+1,j+1] = aaI + dzeta * Ka * (-bbR * ffI - bbI * ffR)
    
                fR[i+1,j+1] = (g * aR[i+1,j+1] * (bbR + dbR) + g * (aI[i+1,j+1]) * (bbI + dbI) + ffR*kappa/dzeta)/(1 + kappa/dzeta)
                fI[i+1,j+1] = (g * aI[i+1,j+1] * (bbR + dbR) - g * (aR[i+1,j+1]) * (bbI + dbI) + ffI*kappa/dzeta)/(1 + kappa/dzeta)
            else:
                daR1 = dzeta * Ka * (bbI * ffI - bbR * ffR)
                daI1 = dzeta * Ka * (-bbR * ffI - bbI * ffR)
                dfR1 = dzeta / kappa * (g * aaR * bbR + g * aaI * bbI - ffR)
                dfI1 = dzeta / kappa * (g * aaI * bbR - g * aaR * bbI - ffI)
    
                daR2 = dzeta * Ka * ((bbI + dbI/2) * (ffI + dfI1/2) - (bbR + dbR/2) * (ffR + dfR1/2))
                daI2 = dzeta * Ka * (-(bbR + dbR/2) * (ffI + dfI1/2) - (bbI + dbI/2) * (ffR + dfR1/2))
                dfR2 = dzeta / kappa * (g * (aaR + daR1/2) * (bbR + dbR/2) + g * (aaI + daI1/2) * (bbI + dbI/2) - (ffR + dfR1/2))
                dfI2 = dzeta / kappa * (g * (aaI + daI1/2) * (bbR + dbR/2) - g * (aaR + daR1/2) * (bbI + dbI/2) - (ffI + dfI1/2))
    
                daR3 = dzeta * Ka * ((bbI + dbI/2) * (ffI + dfI2/2) - (bbR + dbR/2) * (ffR + dfR2/2))
                daI3 = dzeta * Ka * (-(bbR + dbR/2) * (ffI + dfI2/2) - (bbI + dbI/2) * (ffR + dfR2/2))
                dfR3 = dzeta / kappa * (g * (aaR + daR2/2) * (bbR + dbR/2) + g * (aaI + daI2/2) * (bbI + dbI/2) - (ffR + dfR2/2))
                dfI3 = dzeta / kappa * (g * (aaI + daI2/2) * (bbR + dbR/2) - g * (aaR + daR2/2) * (bbI + dbI/2) - (ffI + dfI2/2))
    
                daR4 = dzeta * Ka * ((bbI + dbI) * (ffI + dfI3) - (bbR + dbR) * (ffR + dfR3))
                daI4 = dzeta * Ka * (-(bbR + dbR) * (ffI + dfI3) - (bbI + dbI) * (ffR + dfR3))
                dfR4 = dzeta / kappa * (g * (aaR + daR3) * (bbR + dbR) + g * (aaI + daI3) * (bbI + dbI) - (ffR + dfR3))
                dfI4 = dzeta / kappa * (g * (aaI + daI3) * (bbR + dbR) - g * (aaR + daR3) * (bbI + dbI) - (ffI + dfI3))
    
                aR[i+1,j+1] = aaR + (daR1 + 2*daR2 + 2*daR3 + daR4)/6
                aI[i+1,j+1] = aaI + (daI1 + 2*daI2 + 2*daI3 + daI4)/6
                fR[i+1,j+1] = ffR + (dfR1 + 2*dfR2 + 2*dfR3 + dfR4)/6
                fI[i+1,j+1] = ffI + (dfI1 + 2*dfI2 + 2*dfI3 + dfI4)/6
                
    
        if i+2 < Ntau:
            bR[i+2,:] = bR[i+1,:] + dtau * Kb * (aR[i+1,:] * fR[i+1,:] + aI[i+1,:] * fI[i+1,:])
            bI[i+2,:] = bI[i+1,:] + dtau * Kb * (-aR[i+1,:] * fI[i+1,:] + aI[i+1,:] * fR[i+1,:])
                
    phi_a = np.arctan2(aI,aR)
    phi_b = np.arctan2(bI,bR)
    phi_f = np.arctan2(fI,fR)
    
    a = np.sqrt(aR**2 + aI**2)
    b = np.sqrt(bR**2 + bI**2)
    f = np.sqrt(fR**2 + fI**2)


    return a, b, f, phi_a, phi_b, phi_f

#Example usage------------------------------------------------------------------------------------------------
a, b, f, phi_a, phi_b, phi_f = finite_damp_phase_dependent()

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
