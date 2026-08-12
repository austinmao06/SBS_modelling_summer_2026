<h1>Austin's IFE 2026 Summer Repository</h1>


This repository is a collection of the code I developed for my summer 2026 IFE project at Cornell University. You will find the different integrators used and some functions for efficiency, pulse width, peak intensity, etc.


<h2>The Unprocessed Equations (Damzens):</h2>

$$\frac{\partial E_a}{\partial z} + \frac{n}{c} \frac{\partial E_a}{\partial t} - \frac{i}{2k_a} \nabla_T^2 E_a = \frac{i\omega_a \gamma_e}{4cn\rho_0} E_b \rho$$

$$-\frac{\partial E_b}{\partial z} + \frac{n}{c} \frac{\partial E_b}{\partial t} + \frac{i}{2k_b} \nabla_T^2 E_b = \frac{i\omega_b \gamma_e}{4cn\rho_0} E_a \rho^*$$

$$\frac{\partial \rho}{\partial t} + \left( \frac{\Gamma_B}{2} - \frac{i}{2\Omega}(\Omega^2 - \Omega_B^2) \right) \rho = \frac{i\gamma_e \epsilon_0 k_B}{4v} E_a E_b^*$$

**Defining The Constants**

- z - Longitudinal Propagation Coordinate
- t - Temporal Coordinate
- $\epsilon_0$ Permittivity of Free Space
- c - Speed of Light
- v - Speed of Sound in Medium
- n - Refractive Index
- $E_a$ - Pump Field
- $k_a$ - Pump Wavenumber
- $\omega_a$ - Pump Angular Frequency
- $E_b$ - Seed Field
- $k_b$ - Seed Wavenumber
- $\omega_b$ - Seed Angular Frequency
- $\rho$ - Acoustic Density Perturbation
- $\rho_0$ - Equilibrium Medium Density
- $k_B$ - Acoustic Wavenumber
- $\Gamma_B$ - Brillouin Linewidth of Medium
- $\Omega$ - (Pump - Seed) Frequency Difference
- $\Omega_B$ Brillouin Shift Frequency of Medium
- $\gamma$ - Adiabatic Index
- $\gamma_e$ - Electrostriction Coefficient
- $I_0 = c\epsilon_0 E_0^2$
- $^*$ Denotes Complex Conjugate
