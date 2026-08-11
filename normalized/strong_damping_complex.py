"""
allow for complex envelopes
a = Ae^iphi_a etc...
"""
#SETUP---------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dtau = 0.02
dzeta = 0.025
tau_range = [0, 20]
zeta_range = [-20, 8]
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

a0 = 1
b0 = 1
sigma = 1
g = 1
phi_a_initial = 0
phi_b_initial = 0
#INTEGRATOR FUNCTION------------------------------------------------------------------------------------------
def phase_2wave(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, g=g,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)

    A = np.empty((Ntau, Nzeta))
    B = np.empty((Ntau, Nzeta))
    phi_a = np.empty((Ntau, Nzeta))
    phi_b = np.empty((Ntau, Nzeta))

    A[0,:] = a0
    A[:,0] = a0
    B[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**2)   # seed profile

    for i in range(Ntau): #Inserting arbitrary phases
        phi_a[i,:]  = np.pi*tau[i]/6
    for j in range(Nzeta): #
        phi_b[:,j] = np.pi*zeta[j]*1.5

    # pump pre-march at tau=0 (RK4 in zeta)
    for j in range(Nzeta - 1):
        aa = A[0,j]
        bb = B[0,j]
        db = B[0,j+1] - bb
        ka1 = -dzeta * g * aa * ( bb**2 )
        ka2 = -dzeta * g * (aa + ka1/2) * (bb + db/2)**2
        ka3 = -dzeta * g * (aa + ka2/2) * (bb + db/2)**2
        ka4 = -dzeta * g * (aa + ka3) * (bb + db)**2
        A[0,j+1] = aa + (ka1 + 2*ka2 + 2*ka3 + ka4)/6

    for i in range(Ntau - 1):
        # predictor: Euler in tau, RK4 in zeta
        db1 = dtau * g * B[i,:] * A[i,:]**2
        B[i+1,:] = B[i,:] + db1
        for j in range(Nzeta - 1):
            bb = B[i+1,j]
            db = B[i+1,j+1] - bb
            aa = A[i+1,j]
            da1 = -dzeta * g * aa * bb**2
            da2 = -dzeta * g * (aa + da1/2) * (bb+db/2)**2
            da3 = -dzeta * g * (aa + da2/2) * (bb + db/2)**2
            da4 = -dzeta * g * (aa + da3) * (bb + db)**2
            A[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
        # corrector: full RK4 in tau
        da = A[i+1,:] - A[i,:]
        db1 = dtau * g * B[i,:] * A[i,:]**2
        db2 = dtau * g * (B[i,:] + db1/2) * (A[i,:] + da/2)**2
        db3 = dtau * g * (B[i,:] + db2/2) * (A[i,:] + da/2)**2
        db4 = dtau * g * (B[i,:] + db3) * (A[i,:] + da)**2
        B[i+1,:] = B[i,:] + (db1 + 2*db2 + 2*db3 + db4)/6
        for j in range(Nzeta - 1):
            bb = B[i+1,j]
            db = B[i+1,j+1] - bb
            aa = A[i+1,j]
            da1 = -dzeta * g * aa * bb**2
            da2 = -dzeta * g * (aa + da1/2) * (bb+db/2)**2
            da3 = -dzeta * g * (aa + da2/2) * (bb + db/2)**2
            da4 = -dzeta * g * (aa + da3) * (bb + db)**2
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
