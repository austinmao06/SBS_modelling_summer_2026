#Computing and mounting the necessary constants------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

G_b = 2.0974e9      ###             ###                          #Hz. Brillouin Linewidth of Kr gas
f_B = 1.5075e9                   ###            ###              #Hz. Frequency of Brillouin Shift.
rho_0 = 3.74    ###               ###                  ###       #kg/m3. Equilibrium gas density.
n = 1.000495        ##        ##          ##            ##       #n/a. refractive index of Kr gas.
gamma = 5/3      ###                  ###                        #n/a. Adiabatic index

mu_0 = 4*np.pi*1e-7               ###                ###         #vacuum permeability
ep_0 = 8.854e-12          ###                    ###             #vacuum permittivity
c_light = 1/np.sqrt(mu_0 * ep_0) ###             ###             #m/s
rho_bar = 1.e-4        ###       ###          ###         ###    #n/a. small ad-hoc density perturbation scale.
gamma_e = 2*(n-1)    ###         ###            ###              #n/a. Electrostriction coefficient
P = 101325         ###                          ###              #kg m-1 s-2. Gas pressure
c_sound = np.sqrt(gamma*P/rho_0)      ###             ###        #m/s. Speed of sound in medium.
lambdaL = 248*1e-9        ###                ###                 #m. wavelength of laser light. 
f_laser = c_light/lambdaL      ###                   ###         #Hz. Frequency of laser light. 
w_laser = 2*np.pi*f_laser                ###               ###   #rad/s. Angular frequency of laser light. 
w_B = 2*np.pi*f_B                   ###            ###           #rad/s. Angular Frequency of Brillouin Shift.
K_B = w_B/c_sound      ###            ###                        #rad/m. Wavenumber of acoustic wave.
I_0 = 1e12    ###               ###               ###            #W/m2. Intensity of pump laser.
kNorm = rho_bar*gamma_e*w_laser/(4*c_light*n)                    #rad/m. normalization wavenumber.
wNorm = c_light * kNorm/n          ###             ###           #rad/s. Normalization frequency.
g = gamma_e * K_B * I_0 / (2 *G_b*c_light*c_sound*rho_0*rho_bar) #rad. gain coefficient
#I_0 = c_light*ep_0*E0**2
G = G_b / (2 * wNorm)                                            #1/rad. Normalized damping coefficient.
print(f"c_light = {c_light/1e8}*10^8m/s")
print(f"c_sound = {c_sound}m/s")
print(f"g = {g}")
print(f"G = {G}")
print(f"w_laser = {w_laser/1e15}*10^15Hz")
print(f"wNorm = {wNorm}Hz")
print(f"k = {kNorm}")
#Setup------------------------------------------------------------------------------------------
# FINITE DAMPING case  -- 3wave structure: forward Euler on a, backward Euler on f (overdamped),
#                          RK4 on (a,f) when 2G<1, forward Euler on b.
#   da/dzeta = -(k/(1+n)) * b * f            ,  Ka = k/(1+n)
#   db/dtau  =  (C1*c_light/n) * a * f        ,  Kb = C1*c_light/n
#   kappa*df/dzeta + f = C2*a*b               ,  kappa = 2*c_light/G_b
def finite_damp(zeta_range=zeta_range, dzeta=dzeta, tau_range=tau_range, dtau=dtau,
                a0=a0, b0=b0, sigma=sigma, b_exp=1,
                k=kNorm, g=g, n=n, c_light=c_light, G_b=G_b, G=G):
    tau  = np.arange(tau_range[0], tau_range[1], dtau)
    zeta = np.arange(zeta_range[0], zeta_range[1], dzeta)
    Ntau, Nzeta = len(tau), len(zeta)
    a = np.empty((Ntau, Nzeta))
    b = np.empty((Ntau, Nzeta))
    f = np.empty((Ntau, Nzeta))

    Ka    = k/2
    Kb    = k*c_light/n
    kappa = 2*c_light/(G_b*n)

    # ICs: far-left for a and f, initial profile for b
    b[0,:] = b0 * np.exp(-np.abs(zeta/sigma)**b_exp)
    a[:,0] = a0
    f[:,0] = 0

    # row-0 zeta march: forward Euler on a, backward Euler on f
    
    for j in range(Nzeta - 1):
        a[0,j+1] = a[0,j] - dzeta * Ka * b[0,j] * f[0,j]
        f[0,j+1] = (g*a[0,j+1]*b[0,j+1] + (kappa/dzeta)*f[0,j]) / (1 + kappa/dzeta)
    

    # b kickstart: forward Euler   (db/dtau = Kb * a * f)
    b[1,:] = b[0,:] + dtau * Kb * a[0,:] * f[0,:]

    for i in range(Ntau - 1):
        for j in range(Nzeta - 1):
            bb = b[i+1,j]; db = b[i+1,j+1] - bb
            aa = a[i+1,j]; ff = f[i+1,j]
            if 2*G > 1:
                a[i+1,j+1] = aa - dzeta * Ka * bb * ff
                f[i+1,j+1] = (g*a[i+1,j+1]*(bb+db) + (kappa/dzeta)*ff) / (1 + kappa/dzeta)
            else:
                da1 = -dzeta * Ka * bb * ff
                df1 =  dzeta * (1/kappa) * (g*aa*bb - ff)
                da2 = -dzeta * Ka * (bb + db/2) * (ff + df1/2)
                df2 =  dzeta * (1/kappa) * (g*(aa + da1/2)*(bb + db/2) - (ff + df1/2))
                da3 = -dzeta * Ka * (bb + db/2) * (ff + df2/2)
                df3 =  dzeta * (1/kappa) * (g*(aa + da2/2)*(bb + db/2) - (ff + df2/2))
                da4 = -dzeta * Ka * (bb + db) * (ff + df3)
                df4 =  dzeta * (1/kappa) * (g*(aa + da3)*(bb + db) - (ff + df3))
                a[i+1,j+1] = aa + (da1 + 2*da2 + 2*da3 + da4)/6
                f[i+1,j+1] = ff + (df1 + 2*df2 + 2*df3 + df4)/6
        if i+2 < Ntau:
            b[i+2,:] = b[i+1,:] + dtau * Kb * a[i+1,:] * f[i+1,:]

    print(np.any(np.isinf(b)), np.any(np.isnan(b)))
    return a, b, f
#Example usage------------------------------------------------------------------------------------------------
a, b, f = finite_damp()

#Plotting with Physical Units--------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- unit conversions -------------------------------------------------
field_norm = 100 / 1000          # your existing |a|^2,|b|^2 -> GW/cm^2 factor
                          # (replace with the actual E0-based factor if different)
u_ac  = c_sound**2 * rho_0 * rho_bar**2 / 2.0   # |f|^2 -> J/m^3
ac_norm = 1.0 * 1000

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()                      # pump, same optical units
ax3 = ax1.twinx()                      # acoustic, its own units
ax3.spines['right'].set_position(('outward', 55))   # offset the 3rd axis

def plot_set(t_idx, color, label):
    ax1.plot(zeta, np.abs(b[t_idx, :])**2 * field_norm, color=color,
             linestyle="-",  label=label)
    ax2.plot(zeta, np.abs(a[t_idx, :])**2 * field_norm, color=color, linestyle="--")
    ax3.plot(zeta, np.abs(f[t_idx, :])**2 * u_ac * ac_norm,  color=color, linestyle=":")

t1 = 40e-9
time1 = np.argmin(np.abs(tau - t1))
plot_set(time1, "red",  "10m")

t2 = 80e-9
time1 = np.argmin(np.abs(tau - t2))
plot_set(time1, "green",  "25m")

t3 = 126.7e-9
time3 = np.argmin(np.abs(tau - t3))
plot_set(time3, "blue", "38m")

ax1.set_xlabel(r"$\zeta$ (m)")
ax1.set_ylabel(r"Seed intensity (GW/cm$^2$)")
ax2.set_ylabel(r"Pump intensity (GW/cm$^2$)")
ax3.set_ylabel(r"Acoustic energy density (mJ/m$^3$)")

style_legend = [
    Line2D([0], [0], color="black", linestyle="-",  label=r"seed"),
    Line2D([0], [0], color="black", linestyle="--", label=r"pump"),
    Line2D([0], [0], color="black", linestyle=":",  label=r"acoustic"),
]
color_legend = ax1.get_legend_handles_labels()[0]
ax1.legend(handles=color_legend + style_legend, loc="best")

ax1.set_xlim(-10, 1)
left_min, left_max = ax1.get_ylim()
ax2.set_ylim(left_min * 0.2, left_max*0.2)           
ax3.set_ylim(left_min *5, left_max*5)

plt.title("Finite Damping Amplification (Gaussian Seed Profile)")
plt.tight_layout()
ax1.grid()
plt.show()
