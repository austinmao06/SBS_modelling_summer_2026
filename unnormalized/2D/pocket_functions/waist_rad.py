"""
Tracks radius of seed at zeta = 0 over time, compared to analytical expectations FOR pump off, radial GAUSSIAN
"""
idx_z = np.argmin(np.abs(zeta))
waist_rad = []
for t in range(len(times_to_save)):
    I = b[t][idx_z, :]**2                                  # intensity (b is |b|)
    den = np.trapezoid(I * rad, rad)
    num = np.trapezoid(I * rad**3, rad)
    waist_rad.append(np.sqrt(2 * num / den))               # sqrt(2*<r^2>)  <-- was the sqrt/threshold bug
waist_rad = np.array(waist_rad)

# analytic: propagate the input Gaussian+lens (q-parameter), not w0-as-waist
q0 = 1.0 / (-1.0/F - 1j * 2.0/(kL * w0**2))         
z_axis = times_to_save * c_light / n
w = np.array([np.sqrt(-2.0/(kL * np.imag(1.0/(q0 + zz)))) for zz in z_axis])

plt.plot(z_axis, waist_rad, label="Numerical Radius")
plt.plot(z_axis, w, linestyle="--", color="black", label = "Analytical Expectation")
plt.title(f"Beam Radius at Different Propagation Distances: w0 = {w0} z_r = {round(z_R,2)} F = {F}")
plt.xlabel("Propagation Distance (m)")
plt.ylabel("Radius (m)")
plt.legend()
