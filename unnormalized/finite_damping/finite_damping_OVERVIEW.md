# 1D Finite Damping
Here we keep $\Gamma_B$ finite — we do **not** take $\Gamma_B \to \infty$ — so the acoustic wave is no longer the instantaneous product of the pump and seed beating. 
As before, since we are in 1D, we drop the diffraction term $\nabla_T^2$.
We justify this by assuming plane waves in the transverse direction, resulting in only longitudinal profiles.
## The simplifications are as follows:
- $\partial f/\partial\tau = 0$ (finite $\Gamma_B$ retained)
- $\nabla_T^2 = 0$
- $\delta = 0$
- a, b are real
## Starting point
$$2\frac{\partial a}{\partial \zeta} + \frac{n}{c}\frac{\partial a}{\partial \tau} - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

$$\frac{2}{\Gamma_B}\left(\frac{\partial f}{\partial \tau} + \frac{c}{n}\frac{\partial f}{\partial \zeta}\right) + (1-i\delta)f = gab^*$$

## Simplifying with fast depletion approximation, no detuning 

$$\frac{2c}{n\Gamma_B}\frac{\partial f}{\partial \zeta} + f = gab$$

## Final reduced system

$$\boxed{2\frac{\partial a}{\partial \zeta} = -k_{norm}bf}$$

$$\boxed{\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}af}$$

$$\boxed{\frac{2c}{n\Gamma_B}\frac{\partial f}{\partial \zeta} + f = gab}$$

Because $f$ now carries its own dynamics, this is a genuine three-field system rather than the two-wave reduction obtained in the $\Gamma_B \to \infty$ limit.
