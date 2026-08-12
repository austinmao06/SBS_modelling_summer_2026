#tracks central intensity over time. Used for verifying focusing in pump off cases. + compares with analytical waist location
idx_z = np.argmin(np.abs(zeta))
central_intensity = []


for t in range(len(times_to_save)):
    central_intensity.append(b[t][idx_z,0]**2)

central_intensity = np.array(central_intensity)
plt.plot(times_to_save*1e9, central_intensity, label = "Numerical Intensity at Peak")
plt.axvline(time_to_focus * 1e9, linestyle="--", color="black", label = "Analytical Waist Location")
print(times_to_save[np.argmax(central_intensity)]*1e9, "ns", sep = "")
plt.title(f"Central Intensity over Propagation Time: w0 = {w0} z_r = {round(z_R,2)} F = {F}")
plt.xlabel("Time (ns)")
plt.ylabel("Normalized Intensity")
plt.legend()
