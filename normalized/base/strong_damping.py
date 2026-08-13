"""
strong damping forces f = gab
Also, take everything real (so b = b*, etc)
"""
#SETUP---------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dtau = 0.01
dzeta = 0.01
tau_range = [0, 15] #unitless
zeta_range = [-5, 5] #unitless
tau = np.arange(tau_range[0], tau_range[1], dtau)
zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
Ntau = len(tau)
Nzeta = len(zeta)

c_light = 3e8
n = 1.000495
k = 0.6267
w = c_light * k / n

t_true = tau / w                                   #True time (in s)
z_true = (2 * zeta[None, :] - tau[:, None]) / k    # True distance (in m) (Note that it is zeta and tau dependent, so grid changes at each tau)

a0 = 1
b0 = 1
sigma = 1
g = 1

#INTEGRATOR FUNCTION------------------------------------------------------------------------------------------
def strong_damp(zeta_range = zeta_range, dzeta = dzeta, tau_range = tau_range, dtau = dtau, a0 = a0, b0 = b0, sigma = sigma, g = g):
    tau = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau = len(tau)
    Nzeta = len(zeta)
    a = np.empty((Ntau,Nzeta))
    b = np.empty((Ntau,Nzeta))
    #Initial/Boundary Conditions ------------------------------------------------------------------
    a[:,0] = a0 #Undepleted Left Boundary pump
    b[0,:] = b0*np.exp(-np.abs(zeta/sigma)**2) #SEED PROFILE
    #Iteration 0. Adapts pump at tau=0 to current state.
    for j in range(Nzeta - 1):
        aa = a[0,j]
        bb = b[0,j]
        db = b[0,j+1] - bb
        ka1 = -dzeta * g * aa * ( bb**2 )
        ka2 = -dzeta * g * (aa + ka1/2) * (bb + db/2)**2
        ka3 = -dzeta * g * (aa + ka2/2) * (bb + db/2)**2
        ka4 = -dzeta * g * (aa + ka3) * (bb + db)**2
        a[0,j+1] = aa + (ka1 + 2*ka2 + 2*ka3 + ka4)/6
      
    b[1,:] = b[0,:] + dtau * g * b[0,:] * (a[0,:]**2)
  #Loop
    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i,j]
            db = b[i,j+1] - bb
            aa = a[i,j]
    
            da1 = -dzeta * g * aa * bb**2
            da2 = -dzeta * g * (aa + da1/2) * (bb + db/2)**2
            da3 = -dzeta * g * (aa + da2/2) * (bb + db/2)**2
            da4 = -dzeta * g * (aa + da3) * (bb + db)**2
    
            a[i+1,j+1] = a[i+1,j] + (da1 + 2*da2 + 2*da3 + da4)/6
            if a[i+1,j+1] < 0: a[i+1,j+1] = 0
        if i + 2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * g * b[i+1,:] * (a[i+1,:]**2)
        
    blown = np.where(np.isinf(b))
    print(np.any(np.isinf(b)))
    print(np.any(np.isnan(b)))
    if len(blown[0]) > 0:
        print(f"First blowup at tau index {blown[0].min()}, tau = {tau[blown[0].min()]:.2f}")

    return a, b
#Usage---------------------------------------------------------------------------------------------------------
a, b = strong_damp()
#Plotting (Seed Frame)------------------------------------------------------------------------
t1 = 0
time_1 = np.argmin(np.abs(tau - t1))
plt.plot(zeta, b[time_1,:]**2, color="red", label=f"τ = {t1}")
f = g*a[time_1,:] * b[time_1,:]
#plt.plot(zeta,f**2, color="red", linestyle=":")
#plt.plot(zeta, a[time_1,:]**2, color="red", linestyle="--")

t2 = 2
time_2 = np.argmin(np.abs(tau - t2))
plt.plot(zeta, b[time_2,:]**2, color="green", label=f"τ = {t2}")
f = g*a[time_2,:] * b[time_2,:]
#plt.plot(zeta,f**2, color="green", linestyle=":")
#plt.plot(zeta, a[time_2,:]**2, color="green", linestyle="--")

t3 = 5
time_3 = np.argmin(np.abs(tau - t3))
plt.plot(zeta, b[time_3,:]**2, color="blue", label=f"t = {t3}")
f = g*a[time_3,:] * b[time_3,:]
#plt.plot(zeta,f**2, color="blue", linestyle=":")
#plt.plot(zeta, a[time_3,:]**2, color="blue", linestyle="--")

############################################
"""
#Gaussian Asymptote
x1 = -a0*sigma*(g*t1)**.5
y1 = 2*a0 *(sigma**-1)*(t1/g)**.5

x2 = -a0*sigma*(g*t2)**.5
y2 = 2*a0 *(sigma**-1)*(t2/g)**.5

x3 = -a0*sigma*(g*t3)**.5
y3 = 2*a0 *(sigma**-1)*(t3/g)**.5

limit_x = -a0*sigma*(g*tau)**.5
limit_y = 2*a0 *(sigma**-1)*(tau/g)**.5
"""

"""
#Supergaussian Asymptote
x1 = -sigma * (g*a0**2*t1)**.25
y1 = -4*a0/(g**2 * sigma **2) * x1**3
x2 = -sigma * (g*a0**2*t2)**.25
y2 = -4*a0/(g**2 * sigma **2) * x2**3
x3 = -sigma * (g*a0**2*t3)**.25
y3 = -4*a0/(g**2 * sigma **2) * x3**3

limit_x = -sigma * (g*a0**2*tau)**.25
limit_y = -4*a0/(g**2 * sigma **2) * limit_x**3
"""

"""
#Exponential

x1 = -g* a0**2 * sigma * t1
y1 = (g*sigma)**-1
x2 = -g* a0**2 * sigma * t2
y2 = (g*sigma)**-1
x3 = -g* a0**2 * sigma * t3
y3 = (g*sigma)**-1

limit_x = -g* a0**2 * sigma * tau
john = np.linspace (1,2,len(limit_x))
limit_y = (g*sigma)**-1 * john/john
"""

plt.xlim(left=-3,right=2)
#plt.ylim(bottom = 0,top=3)
plt.title("Gaussian Seed Amplification (Artificial Parameters)")
plt.legend()
plt.grid()
plt.xlabel("ζ")
plt.ylabel(r"$b^2$")

#Plotting (Lab Frame)------------------------------------------------------------------------
plt.plot(z_true[time_1,:], b[time_1,:]**2, color="red", label=f"t = {t1/w * 1e9}ns")
plt.plot(z_true[time_2,:], b[time_2,:]**2, color="green", label=f"t = {t2/w * 1e9}ns")
plt.plot(z_true[time_3,:], b[time_3,:]**2, color="blue", label=f"t = {t3/w * 1e9}ns")
plt.xlabel("z (m)")
plt.ylabel(r"$b^2$")
plt.legend()
plt.title("Lab Frame")
