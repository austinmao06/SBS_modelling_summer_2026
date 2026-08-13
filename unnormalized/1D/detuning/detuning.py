#Computing and mounting the necessary constants------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

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

#Set Detuning

dlambda = 1e-12 ####                                           # for detuning laser wavelength diff
Nlinewidths = 1.5####                                            #how many G_b from f_B is Omega
#Omega = c_light/(lambdaL**2) * dlambda                         #Hz. Translated frequency difference from dlambda
Omega = OmegaB + Nlinewidths *GammaB                            #Hz. Translated frequency difference from dlambda
d = (Omega**2 - OmegaB**2)/(GammaB*Omega)                         #d from physical parameters
#d = 2                                                         #d from raw

print(f"c_light = {c_light/1e8}*10^8m/s")
print(f"c_sound = {c_sound}m/s")
print(f"g = {g}")
print(f"G = {G}")
print(f"w_laser = {wL/1e15}*10^15Hz")
print(f"wNorm = {wNorm}Hz")
print(f"k = {kNorm}")
print(f"w_B = {wB}")
print(f"d = {d}")
#Natural Parameters of detuning ------------------------------------------------------------------------------------------
true_beat = (GammaB*d + np.sqrt(GammaB**2 * d**2 + 4*OmegaB**2))/2
dlambda = lambdaL**2 / c_light * true_beat
Nlinewidths = (true_beat - OmegaB)/GammaB

print(f"Omega = {true_beat/1e9} GHz")
print(f"Dlambda = {round(dlambda*1e12,2)} pm")
print(f"Omega = {OmegaB} + {Nlinewidths} GammaB")
#Setup------------------------------------------------------------------------------------------

dzeta = 0.01                       # zeta resolution (m)
dtau = 0.8 * dzeta / c_light       # CFL-style: 0.8 * (dzeta / c). tau is TRUE time (s); c/n≈c since n≈1
tau_range = [0, 127e-9]
zeta_range = [-10, 5]
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

t_true = tau                                   # physical time per tau index (s)
z_true = (zeta[None, :] - c_light*tau[:, None]/n)   # shape (Ntau, Nzeta), lab position (m)

a0 = 1
b0 = 1
sigma = 0.234   # chosen so pulse_duration(t=0) ≈ 0.3 m (lab-frame length metric)

phi_a_initial = 0
phi_b_initial = 0
#Integrator Function--------------------------------------------------------------------------------
def detune(d=d, zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
           a0=a0, b0=b0, sigma=sigma, g=g, G=G, b_exp=1,
           phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    import numpy as np
    import matplotlib.pyplot as plt
    #PARAMS####################################################
    a0 = a0
    g = g
    sigma = sigma
    b0 = b0
    ########################################################
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    Ka = kNorm/2
    Kb = kNorm*c_light/n
    kappa = 2*c_light/(GammaB*n)
    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)
    f = np.empty((Ntau, Nzeta), dtype=complex)
    #####################ICs#############################
    """
    a, f demand far-left limits.
    b requires initial conditions
    """
    a[:,0] = a0 * np.exp(1j*phi_a_initial)              # constant pump phase at entrance
    b0_profile = b0 * np.exp(-np.abs(zeta/sigma)**b_exp)
    b[0,:] = b0_profile * np.exp(1j*phi_b_initial)
    f[:,0] = 0                                          # far-left column: no incoming acoustic

    den = (1 + kappa/dzeta) - 1j*d                      # backward-Euler phonon: kappa*d_zeta f + (1 - i d) f = g a b*

    # t = 0 row: march a, f in zeta
    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] + dzeta * Ka * (-b[0,j] * f[0,j])
        f[0,j+1] = (g * a[0,j+1] * np.conj(b[0,j+1]) + kappa/dzeta * f[0,j]) / den

    b[1,:] = b[0,:] + dtau * Kb * (a[0,:] * np.conj(f[0,:]))

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]
            db = b[i+1,j+1] - bb
            aa = a[i+1,j]
            ff = f[i+1,j]
            if 2*G>1:
                a[i+1,j+1] = aa + dzeta * Ka * (-bb * ff)
                f[i+1,j+1] = (g * a[i+1,j+1] * np.conj(b[i+1,j+1]) + kappa/dzeta * f[i+1,j]) / den
            else:
                da1 = dzeta * Ka * (-bb * ff)
                df1 = dzeta/kappa * (g * aa * np.conj(bb) - (1 - 1j*d) * ff)

                da2 = dzeta * Ka * (-(bb + db/2) * (ff + df1/2))
                df2 = dzeta/kappa * (g * (aa + da1/2) * np.conj(bb + db/2) - (1 - 1j*d) * (ff + df1/2))

                da3 = dzeta * Ka * (-(bb + db/2) * (ff + df2/2))
                df3 = dzeta/kappa * (g * (aa + da2/2) * np.conj(bb + db/2) - (1 - 1j*d) * (ff + df2/2))

                da4 = dzeta * Ka * (-(bb + db) * (ff + df3))
                df4 = dzeta/kappa * (g * (aa + da3) * np.conj(bb + db) - (1 - 1j*d) * (ff + df3))

                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6

        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * (a[i+1,:] * np.conj(f[i+1,:]))

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    phi_f = np.angle(f)

    a = np.abs(a)
    b = np.abs(b)
    f = np.abs(f)

    print(a.shape)
    print(b.shape)
    print(np.any(np.isinf(b)))
    print(np.any(np.isnan(b)))
    # find where blowup first occurs
    blown = np.where(np.isinf(b))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    return a, b, f, phi_a, phi_b, phi_f

#example usage
a, b, f, *_ = detune()
