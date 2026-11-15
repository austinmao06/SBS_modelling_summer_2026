#SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

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

#Params
dzeta = 0.01                    # zeta resolution (m)
dtau = 0.8 * dzeta / c_light       # CFL-style: 0.8 * (dzeta / c). tau is TRUE time (s); c/n≈c since n≈1
tau_range = [0, 130e-9]
zeta_range = [-10, 5]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index (s)
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta), lab position (m)
a0 = 1
b0 = 1
sigma = 1
b_exp = 2

times_to_save = np.arange(0,tau_range[1], 1e-9)
save_indices = sorted({int(np.argmin(np.abs(tau - t))) for t in times_to_save})
save_length = len(save_indices)


theta=1-1/np.sqrt(2)
eta = 1-1/(2*theta)

Tab_L = np.array([[0,0,0],[0,theta,0], [0,1-theta,theta]])
#Tab_L_weight = np.array([0,1-theta,theta])
Tab_N = np.array([[0,0,0],[theta,0,0],[eta,1-eta,0]])
#Tab_N_weight = np.array([eta,1-eta,0])

#integrator
def full_form(d=d, zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
           a0=a0, b0=b0, sigma=sigma, g=g, G=G, b_exp=b_exp,
           phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial, times_to_save=times_to_save, 
              save_indices = save_indices, save_length = save_length):
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
    a = np.empty((save_length, Nzeta), dtype=complex)
    a[0,:] = a0
    b = np.empty((save_length, Nzeta), dtype=complex)
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp)
    f = np.empty((save_length, Nzeta), dtype=complex)
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
        V[0] = a0 #undepleted pump
        V[2*Nzeta] = 0 #no acoustic
        return V
    #Construct L Matrix______________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    aL1 = np.ones(Nzeta) * 2*c_light/(n*2*dzeta)
    aL1[0] = 0; aL1[-1] = 2*c_light/(n*dzeta)
    aL2 = np.zeros(Nzeta)
    aL2[0] = 0; aL2[-1] = -2*c_light/(n*dzeta)
    aL3 = np.ones(Nzeta) * -2*c_light/(n*2*dzeta)
    aL3[0] = 0; aL3[-1] = 0

    bL1 = np.zeros(Nzeta); bL2 = np.zeros(Nzeta); bL3 = np.zeros(Nzeta)

    fL1 = np.ones(Nzeta) *c_light/(n*2*dzeta)
    fL1[0] = 0; fL1[-1] = c_light/(n*dzeta)
    fL2 = np.ones(Nzeta) * -GammaB * (1-(1j)*d)/2
    fL2[0] = 0; fL2[-1] = -c_light/(n*dzeta) - GammaB * (1-(1j)*d)/2
    fL3 = np.ones(Nzeta) * -c_light/(n*2*dzeta)
    fL3[0] = 0; fL3[-1] = 0

    L1 = np.concatenate((aL1, bL1, fL1)); L2 = np.concatenate((aL2, bL2, fL2)); L3 = np.concatenate((aL3, bL3, fL3))
    LM = np.zeros((3, 3*Nzeta), dtype = complex)
    LM[0,1:] = L3[:-1]; LM[1,:] = L2; LM[2,:-1] = L1[1:]
    
    solve_M = -dtau*theta*LM
    solve_M[1, :] = solve_M[1, :] + 1
    L = dia_array((LM, [1,0,-1]), shape=(3*Nzeta, 3*Nzeta))
    #________________________________________________________________________________________________________________
    #________________________________________________________________________________________________________________
    #Integration_____________________________________________________________________________________________________
    
    V0 = np.concatenate((a[0,:], b[0,:],f[0,:]))
    next_save = 1
    for j in range(Ntau-1):
        N1 = N(V0)
        RS = BC(V0 + dtau * Tab_N[1,0]*N1)
        V1 = solve_banded((1,1),solve_M, RS)
        N2 = N(V1)
        LV1 = L@V1
        RS = BC(V0 + dtau * (Tab_L[2,1]*LV1 + Tab_N[2,0]*N1 + Tab_N[2,1]*N2))
        V2 = solve_banded((1,1), solve_M, RS)
        V0 = V2
        if next_save <save_length and j == save_indices[next_save]:
            a[next_save,:] = V0[:Nzeta]; b[next_save,:] = V0[Nzeta:2*Nzeta]; f[next_save,:] = V0[2*Nzeta:3*Nzeta]
            next_save +=1
    

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

    return a, b, f, phi_a, phi_b, phi_f






