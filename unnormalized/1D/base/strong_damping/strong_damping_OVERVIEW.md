# 1D Strong Damping

This is the starting point — we take $\Gamma_B \to \infty$ while keeping $g$ finite. This results in an acoustic wave that is the immediate product of the pump and seed beating (infinite damping).

Thus, the equations devolve into a two-wave system, much simpler to deal with. Also, since we are in 1D, we drop the diffraction term $\nabla_T^2$.

We justify this by assuming plane waves in the transverse direction, resulting in only longitudinal profiles.

## The simplifications are as follows:

- $\Gamma_B \to \infty$
- $\nabla_T^2 = 0$
- $\delta = 0$
- a, b are real

## Starting point

$$2\frac{\partial a}{\partial \zeta} + \frac{n}{c}\frac{\partial a}{\partial \tau} - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

$$\frac{2}{\Gamma_B}\left(\frac{\partial f}{\partial \tau} + \frac{c}{n}\frac{\partial f}{\partial \zeta}\right) + (1-i\delta)f = gab^*$$

## Simplifying

$$(1-i\delta)f = gab^* \quad\longrightarrow\quad f = gab^*$$

With a, b real:

$$f = gab$$

$$2\frac{\partial a}{\partial \zeta} = -k_{norm}bf$$

$$2\frac{\partial a}{\partial \zeta} = -k_{norm}gab^2$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}af^*$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}ga^2b$$

## Final reduced 2-wave system

$$\boxed{2\frac{\partial a}{\partial \zeta} = -k_{norm}gab^2}$$

$$\boxed{\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}ga^2b}$$
