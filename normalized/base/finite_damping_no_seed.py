"""
No initial seed. Instead create an initial acoustic wave.
Complex envelopes: a, b, f are complex arrays (initial phases set by phi_a_initial, phi_f_initial)
"""
#setup
import numpy as np
import matplotlib.pyplot as plt
#############################################################################

#BASE SETUP##################################################################
dzeta = 0.0025
dtau = 0.0025

tau_range = [0, 20]
zeta_range = [-15, 5]

tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)

Ntau = len(tau)
Nzeta = len(zeta)

c_light = 3e8
n = 1.000495
k = 0.1
w = c_light * k / n

t_true = tau / w                                   # physical time per tau index
z_true = (2 * zeta[None, :] - tau[:, None]) / k    # shape (Ntau, Nzeta)
#############################################################################

#Longitudinal Pulse Parameters ##############################################
a0 = 1
f0 = 0.22
f_exp = 1
sigma = 1 #initial acoustic longitudinal width
phi_a_initial = 0 #initial pump phase
phi_f_initial = 0 #initial acoustic phase
g = 1
G = 100000
#############################################################################

#integrator
def seedless(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
             a0=a0, f0=f0, sigma=sigma, f_exp=f_exp, g=g, G=G,
             phi_a_initial=phi_a_initial, phi_f_initial=phi_f_initial):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)
    f = np.empty((Ntau, Nzeta), dtype=complex)

    b[0,:] = 0                                                                  # seedless: no initial seed
    a[:,0] = a0 * np.exp(1j*phi_a_initial)                                      # constant pump source
    f[:,0] = 0                                                                  # far-left limit of no acoustic
    f[0,:] = f0 * np.exp(-np.abs(zeta/sigma)**f_exp) * np.exp(1j*phi_f_initial) # initial acoustic profile

    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] - dzeta * (b[0,j] * f[0,j])

    b[1,:] = b[0,:] + dtau * a[0,:] * np.conj(f[0,:])

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]
            db = b[i+1,j+1] - bb
            aa = a[i+1,j]
            ff = f[i+1,j]

            if 2*G > 1:
                a[i+1, j+1] = aa - dzeta * bb * ff
                f[i+1,j+1] = (g * a[i+1,j+1] * np.conj(bb+db) + ff/(2*G*dzeta) )/(1 + 1/(2*G*dzeta))
            else:
                da1 = -dzeta * bb * ff
                df1 = 2 * G * dzeta * (g * aa * np.conj(bb) - ff)

                da2 = -dzeta * (bb + db/2) * (ff + df1/2)
                df2 = 2 * G * dzeta * (g * (aa + da1/2) * np.conj(bb + db/2) - (ff + df1/2))

                da3 = -dzeta * (bb + db/2) * (ff + df2/2)
                df3 = 2 * G * dzeta * (g * (aa + da2/2) * np.conj(bb + db/2) - (ff + df2/2))

                da4 = -dzeta * (bb + db) * (ff + df3)
                df4 = 2 * G * dzeta * (g * (aa + da3) * np.conj(bb + db) - (ff + df3))

                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6

        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * a[i+1,:] * np.conj(f[i+1,:])

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    phi_f = np.angle(f)

    print(np.any(np.isinf(b)))
    print(np.any(np.isnan(b)))
    blown = np.where(np.isinf(b))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    return a, b, f, phi_a, phi_b, phi_f
