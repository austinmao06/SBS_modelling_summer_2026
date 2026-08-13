"""
allow for complex values
a = aR + iaI, etc
I forgot about dtype complex here....
"""
#SETUP---------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dtau = 0.01
dzeta = 0.01
tau_range = [0, 20]
zeta_range = [-25, 10]
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

a0 = 1.5
b0 = 1
sigma = 1
g = 1
G = 10
phi_a_initial = 0
phi_b_initial = np.pi/2
#INTEGRATOR FUNCTION------------------------------------------------------------------------------------------
def phase_3wave(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, g=g, G=G,
                phi_a_initial=phi_a_initial, phi_b_initial=phi_b_initial):
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)
    
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
    
    g = 1
    G = 10
    a0 = 1
    b0 = 1
    b0_profile = b0 * np.exp(-np.abs(zeta/sigma)**2)
    
    aR[:,0] = a0 * np.cos(phi_a_initial)
    aI[:,0] = a0 * np.sin(phi_a_initial)
    
    bR[0,:] = b0_profile * np.cos(phi_b_initial)
    bI[0,:] = b0_profile * np.sin(phi_b_initial)
    
    fR[:,0] = 0   # far-left column: no incoming acoustic (keep)
    fI[:,0] = 0
    
    for j in range(Nzeta - 1):
        aR[0,j+1] = aR[0,j] + dzeta * (bI[0,j] * fI[0,j] - bR[0,j] * fR[0,j])
        aI[0,j+1] = aI[0,j] + dzeta * (-bR[0,j] * fI[0,j] - bI[0,j] * fR[0,j])
        fR[0,j+1] = (g * aR[0,j+1] * bR[0,j+1] + g * aI[0,j+1] * bI[0,j+1] + fR[0,j]/(2*G*dzeta))/(1 + 1/(2*G*dzeta))
        fI[0,j+1] = (g * aI[0,j+1] * bR[0,j+1] - g * aR[0,j+1] * bI[0,j+1] + fI[0,j]/(2*G*dzeta))/(1 + 1/(2*G*dzeta))
    
    bR[1,:] = bR[0,:] + dtau * (aR[0,:] * fR[0,:] + aI[0,:] * fI[0,:])
    bI[1,:] = bI[0,:] + dtau * (-aR[0,:] * fI[0,:] + aI[0,:] * fR[0,:])
    
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
                aR[i+1,j+1] = aaR + dzeta * (bbI * ffI - bbR * ffR)
                aI[i+1,j+1] = aaI + dzeta * (-bbR * ffI - bbI * ffR)
    
                fR[i+1,j+1] = (g * aR[i+1,j+1] * (bbR + dbR) + g * (aI[i+1,j+1]) * (bbI + dbI) + ffR/(2*G*dzeta))/(1 + 1/(2*G*dzeta))
                fI[i+1,j+1] = (g * aI[i+1,j+1] * (bbR + dbR) - g * (aR[i+1,j+1]) * (bbI + dbI) + ffI/(2*G*dzeta))/(1 + 1/(2*G*dzeta))
            else:
                daR1 = dzeta * (bbI * ffI - bbR * ffR)
                daI1 = dzeta * (-bbR * ffI - bbI * ffR)
                dfR1 = 2 * G * dzeta * (g * aaR * bbR + g * aaI * bbI - ffR)
                dfI1 = 2 * G * dzeta * (g * aaI * bbR - g * aaR * bbI - ffI)
    
                daR2 = dzeta * ((bbI + dbI/2) * (ffI + dfI1/2) - (bbR + dbR/2) * (ffR + dfR1/2))
                daI2 = dzeta * (-(bbR + dbR/2) * (ffI + dfI1/2) - (bbI + dbI/2) * (ffR + dfR1/2))
                dfR2 = 2 * G * dzeta * (g * (aaR + daR1/2) * (bbR + dbR/2) + g * (aaI + daI1/2) * (bbI + dbI/2) - (ffR + dfR1/2))
                dfI2 = 2 * G * dzeta * (g * (aaI + daI1/2) * (bbR + dbR/2) - g * (aaR + daR1/2) * (bbI + dbI/2) - (ffI + dfI1/2))
    
                daR3 = dzeta * ((bbI + dbI/2) * (ffI + dfI2/2) - (bbR + dbR/2) * (ffR + dfR2/2))
                daI3 = dzeta * (-(bbR + dbR/2) * (ffI + dfI2/2) - (bbI + dbI/2) * (ffR + dfR2/2))
                dfR3 = 2 * G * dzeta * (g * (aaR + daR2/2) * (bbR + dbR/2) + g * (aaI + daI2/2) * (bbI + dbI/2) - (ffR + dfR2/2))
                dfI3 = 2 * G * dzeta * (g * (aaI + daI2/2) * (bbR + dbR/2) - g * (aaR + daR2/2) * (bbI + dbI/2) - (ffI + dfI2/2))
    
                daR4 = dzeta * ((bbI + dbI) * (ffI + dfI3) - (bbR + dbR) * (ffR + dfR3))
                daI4 = dzeta * (-(bbR + dbR) * (ffI + dfI3) - (bbI + dbI) * (ffR + dfR3))
                dfR4 = 2 * G * dzeta * (g * (aaR + daR3) * (bbR + dbR) + g * (aaI + daI3) * (bbI + dbI) - (ffR + dfR3))
                dfI4 = 2 * G * dzeta * (g * (aaI + daI3) * (bbR + dbR) - g * (aaR + daR3) * (bbI + dbI) - (ffI + dfI3))
    
                aR[i+1,j+1] = aaR + (daR1 + 2*daR2 + 2*daR3 + daR4)/6
                aI[i+1,j+1] = aaI + (daI1 + 2*daI2 + 2*daI3 + daI4)/6
                fR[i+1,j+1] = ffR + (dfR1 + 2*dfR2 + 2*dfR3 + dfR4)/6
                fI[i+1,j+1] = ffI + (dfI1 + 2*dfI2 + 2*dfI3 + dfI4)/6
                
    
        if i+2 < Ntau:
            bR[i+2,:] = bR[i+1,:] + dtau * (aR[i+1,:] * fR[i+1,:] + aI[i+1,:] * fI[i+1,:])
            bI[i+2,:] = bI[i+1,:] + dtau * (-aR[i+1,:] * fI[i+1,:] + aI[i+1,:] * fR[i+1,:])
                
    phi_a = np.arctan2(aI,aR)
    phi_b = np.arctan2(bI,bR)
    phi_f = np.arctan2(fI,fR)
    
    a = np.sqrt(aR**2 + aI**2)
    b = np.sqrt(bR**2 + bI**2)
    f = np.sqrt(fR**2 + fI**2)


    return a, b, f, phi_a, phi_b, phi_f
#Usage--------------------------------------------
curves = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
phi_as = [0,0,0,0,np.pi/6,np.pi/4,np.pi/2,2*np.pi/3,np.pi]
phi_bs = [np.pi/2,np.pi/4,np.pi/6,0,0,0,0,0,0]

for i, bn in enumerate(curves):
  bn = phase_3wave(phi_a_initial=phi_as[i], phi_b_initial=phi_bs[i])[1]

t = 20
time = np.argmin(np.abs(tau - t))

slope = 0.75   # horizontal shift per unit vertical offset (4:1 = run:rise)

colors = ["black", "red", "black", "red", "black", "red", "black","red", "black"]

for n, (curve, c) in enumerate(zip(curves, colors), start=1):
    x_offset = slope * n * 2        # horizontal shift grows with stack level
    y_offset = n * 2               # vertical shift (same as before)
    plt.plot(zeta + x_offset, curve[time,:]**2 + y_offset, color=c)

# the reference axvline should slope too, so it stays aligned with the stack
# draw it as a line connecting the shifted baseline positions
ax_x = [-20 + slope*1, -20 + slope*len(curves)*2.4]
ax_y = [1, len(curves)*2.4]
plt.title(f"Comparison of Amplification with Different Phases (τ  = {t})")
#plt.plot(ax_x, ax_y, color="black", linestyle="--")
plt.show()
