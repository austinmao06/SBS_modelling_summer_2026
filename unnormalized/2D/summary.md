<h1>The 2D Case</h1>

In 2D, we approximate the pulses as having radial profiles rather than full plane waves, though maintaining axisymmetry. This is enough to induce diffraction,
which causes focusing. Thus, in 2D, we must, at all times, consider complex-valued envelopes, as phase wavefronts are necessary for focusing.

# Transverse Laplacian in Cylindrical Coordinates

The transverse Laplacian acts in the plane perpendicular to the propagation axis $\zeta$:

$$\nabla_T^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$$

**Change of coordinates.** Switch from Cartesian $(x,y)$ to polar $(r,\theta)$:

$$x = r\cos\theta, \qquad y = r\sin\theta$$

with inverse $r = \sqrt{x^2 + y^2}$ and $\theta = \arctan(y/x)$.

**Chain rule.** Differentiating the inverse relations gives the geometric factors

$$\frac{\partial r}{\partial x} = \cos\theta, \quad \frac{\partial r}{\partial y} = \sin\theta, \quad \frac{\partial \theta}{\partial x} = -\frac{\sin\theta}{r}, \quad \frac{\partial \theta}{\partial y} = \frac{\cos\theta}{r}$$

so the Cartesian derivatives become

$$\frac{\partial}{\partial x} = \cos\theta\frac{\partial}{\partial r} - \frac{\sin\theta}{r}\frac{\partial}{\partial \theta}$$

$$\frac{\partial}{\partial y} = \sin\theta\frac{\partial}{\partial r} + \frac{\cos\theta}{r}\frac{\partial}{\partial \theta}$$

**Swap.** Applying each operator twice and adding, the cross terms cancel and the $\cos^2\theta + \sin^2\theta = 1$ terms combine, leaving

$$\nabla_T^2 = \frac{\partial^2}{\partial r^2} + \frac{1}{r}\frac{\partial}{\partial r} + \frac{1}{r^2}\frac{\partial^2}{\partial \theta^2}$$

# Full Form Equations

Substituting the cylindrical operator into the optical equations (the acoustic equation has no transverse Laplacian):

$$2\frac{\partial a}{\partial \zeta} + \frac{n}{c}\frac{\partial a}{\partial \tau} - \frac{i}{2k_a}\left(\frac{\partial^2 a}{\partial r^2} + \frac{1}{r}\frac{\partial a}{\partial r} + \frac{1}{r^2}\frac{\partial^2 a}{\partial \theta^2}\right) = -k_{norm}bf$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_b}\left(\frac{\partial^2 b}{\partial r^2} + \frac{1}{r}\frac{\partial b}{\partial r} + \frac{1}{r^2}\frac{\partial^2 b}{\partial \theta^2}\right) = k_{norm}af^*$$

$$\frac{2}{\Gamma_B}\left(\frac{\partial f}{\partial \tau} + \frac{c}{n}\frac{\partial f}{\partial \zeta}\right) + (1-i\delta)f = gab^*$$

# Simplifications

- **Fast depletion:** $\frac{\partial}{\partial \zeta} >> \frac{n}{c}\frac{\partial}{\partial \tau}$
- **Axial symmetry:** $\frac {\partial}{\partial \theta}$, so the transverse Laplacian reduces to

$$\nabla_T^2 = \frac{\partial^2}{\partial r^2} + \frac{1}{r}\frac{\partial}{\partial r}$$

- **Pump/Seed Frequency Approximation:** $k_a \approx k_b = k_L$

Applying:

$$\boxed{2\frac{\partial a}{\partial \zeta} - \frac{i}{2k_L}\left(\frac{\partial^2 a}{\partial r^2} + \frac{1}{r}\frac{\partial a}{\partial r}\right) = -k_{norm}bf}$$

$$\boxed{\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_L}\left(\frac{\partial^2 b}{\partial r^2} + \frac{1}{r}\frac{\partial b}{\partial r}\right) = k_{norm}af^*}$$

$$\boxed{\frac{2c}{n\Gamma_B}\frac{\partial f}{\partial \zeta} + (1-i\delta)f = gab^*}$$
