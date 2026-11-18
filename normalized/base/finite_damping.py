"""
Finite Damping allows acoustic relaxation
Complex envelopes: a, b, f are complex arrays (initial phases set by phi_a_initial, phi_b_initial)
"""
#setup
import numpy as np
import matplotlib.pyplot as plt
#############################################################################

#BASE SETUP##################################################################
dzeta = 0.01
dtau = 0.01

tau_range = [0, 30]
zeta_range = [-15, 45]

tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)

Ntau = len(tau)
Nzeta = len(zeta)

c_light = 3e8
n = 1.000495
k = 0.62674
w = c_light * k / n

t_true = tau / w                                   # physical time per tau index
z_true = (2 * zeta[None, :] - tau[:, None]) / k    # shape (Ntau, Nzeta)
#############################################################################

#Longitudinal Pulse Parameters ##############################################
a0 = 1
b0 = 0.22
b_exp = 2
sigma = 0.091 #seed longitudinal gaussian width
phi_a_initial = 0 #initial pump phase
phi_b_initial = 0 #initial seed phase
g = 0.4415
G = 5.5841
#############################################################################

#integrator
def finite_damp(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, b_exp=b_exp, g=g, G=G,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    a = np.empty((Ntau, Nzeta), dtype=complex)
    b = np.empty((Ntau, Nzeta), dtype=complex)
    f = np.empty((Ntau, Nzeta), dtype=complex)
    #Initial Conditions------------------------------------------------------------------------------------------------
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp) * np.exp(1j*phi_b_initial)   # initial seed profile
    a[:,0] = a0 * np.exp(1j*phi_a_initial)                                       # constant pump source
    f[:,0] = 0                                                                   # far-left limit of no acoustic

    for j in range(Nzeta - 1):
        da = - dzeta * b[0,j] * f[0,j]
        a[0,j+1] = a[0,j] + da
        f[0,j+1] = (g * a[0,j+1] * np.conj(b[0,j+1]) + f[0,j]/(2*G*dzeta) )/(1 + 1/(2*G*dzeta))

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
