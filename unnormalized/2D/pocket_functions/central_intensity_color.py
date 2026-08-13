"""
Plots Intensity at each radius of select zeta against propagation distance
"""
z_index = np.argmin(np.abs(zeta-0))
intensities = []
for t in range(len(times_to_save)):
    intensities.append(b[t,z_index,:]**2)
intensities = np.array(intensities).T / 10

def get_waist(b=b):
    idx_z = np.argmin(np.abs(zeta))
    waist_rad = []
    for t in range(len(times_to_save)):
        I = b[t][idx_z, :]**2                                  # intensity (b is |b|)
        den = np.trapezoid(I * rad, rad)
        num = np.trapezoid(I * rad**3, rad)
        waist_rad.append(np.sqrt(2 * num / den))               # sqrt(2*<r^2>)  <-- was the sqrt/threshold bug
    waist_rad = np.array(waist_rad)
    return waist_rad
bwaist = get_waist()


plt.figure()
dist = times_to_save * c_light/n
plt.imshow(intensities, aspect="auto", origin = "lower", extent = [dist[0], dist[-1], rad[0], rad[-1]], cmap = "inferno")
plt.xlabel("Propagation Distance (m)")
plt.ylabel("Radius (m)")
plt.colorbar(label = r"Seed Intensity (GW/$cm^2$)") 
cs = plt.contour(dist, rad, intensities,
                 levels=6, colors='white', linewidths=0.6)
plt.title('Intensity at Seed Peak vs. Propagation Distance')
plt.plot(dist, bwaist, label="Numerical Radius", color="red")
plt.axvline((time_to_focus)*c_light/n, linestyle="--", color="white", label = "Analytical Waist Location (Pure Focusing)")
plt.ylim(top = 0.003)
plt.legend()
plt.show()
