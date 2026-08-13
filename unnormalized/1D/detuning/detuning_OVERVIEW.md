<h1> What is Detuning? </h1>

Detuning is the offset of the true optical beat frequency ($\omega_a - \omega_b$) from the natural Brillouin shift frequency ($\Omega_B$). 
Nonzero detuning equates to an off-resonance drive of the acoustic wave, resulting in a weaker response and hence lessened extent of SBS amplification.

To investigate the effects of detuning, there are a number of ways we can determine the parameters.

$$\delta \equiv \frac{\Omega^2 - \Omega_B^2}{\Gamma_B\Omega}$$ where $\delta$ is the detuning factor

$\Omega = \omega_a - \omega_b$ -- true acoustic driving frequency

$\Omega_B$ -- natural Brillouin Shift Frequency

$\Omega = \frac{c}{\lambda^2}\Delta \lambda$ -- $\Omega$ computed from wavelength difference between pump and seed

Alternatively, we can compute physical scaling from arbitrary detuning factors:

$$\Omega = \frac{\Gamma_B\delta + \sqrt{\Gamma_B^2\delta^2 + 4\Omega_B^2}}{2}$$

$$\Delta\Lambda = \frac{\lambda^2}{2c}\left[\Gamma_B\delta + \sqrt{\Gamma_B^2\delta^2 + 4\Omega_B^2}\right]$$

$$N_{\mathrm{linewidths}} = \frac{\Gamma_B\delta + \sqrt{\Gamma_B^2\delta^2 + 4\Omega_B^2 - 2\Omega_B}}{2\Gamma_B}$$

<h2> Additional Notes</h2>

$$\delta \approx \frac{2\Delta\omega}{\Gamma_B} \quad \text{for } \Omega \approx \Omega_B \text{ (small detuning)}$$

$$\text{where } \Delta\omega = \Omega - \Omega_B.\text{ Note that this approximation isn't usually valid because } \omega_a\, \omega_b >> \Omega$$

Given the real gas parameters, wavelength offsets of the order of 1pm result in detuning factors of 1 or 2. Of course, $\Delta \lambda = 0$ does not correspond to $\delta = 0$.

Studies of detuning are only significant for up to around $\delta = 2$. 
