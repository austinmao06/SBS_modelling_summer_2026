<h1> Unnormalized Transformations</h1>

These are the transformations used for the rest of the code, including detuning, 2D, and a more extensive collection of functions. This transformation was uniquely made, so it is not recorded in  past research of the group.
However, it is simpler in translating back and forth between coordinates, more intuitive, and keeps physical units. This makes setting up simulations much easier, and allows us to better characterize pulses without having to switch frames constantly.
Below is the mathematical derivation for the full-form equations (without any dropped terms, with the exception of $\alpha$ since we are not dealing with STBS). Further equation simplifications will be made in their own files.

Note that the use of kNorm is purely for a simplification of variables, and not for a coordinate transformation here.

## Starting equations (with diffraction and damping term)

$$\frac{dE_a}{dz} + \frac{n}{c}\frac{dE_a}{dt} - \frac{i}{2k_a}\nabla_T^2 E_a + \frac{\alpha}{2}E_a = \frac{i\omega_a\gamma_e}{4cn\rho_0}E_b\rho$$

$$-\frac{\partial E_b}{\partial z} + \frac{n}{c}\frac{\partial E_b}{\partial t} + \frac{i}{2k_b}\nabla_T^2 E_b + \frac{\alpha}{2}E_b = \frac{i\omega_b\gamma_e}{4cn\rho_0}E_a\rho^*$$

$$-2i\Omega\frac{\partial \rho}{\partial t} - \left(i\Omega\Gamma_B + (\Omega^2 - \Omega_B^2)\right)\rho = \frac{\gamma_e\epsilon_0 k_B^2}{2}E_aE_b^*$$

## Assumptions and definitions

$$\alpha \to 0$$

assume $\omega_a \simeq \omega_b$

$$k_{norm} = \frac{\omega_a\gamma_e\tilde\rho_0}{4cn}$$

$$g = \frac{\gamma_e\epsilon_0 ck_B E_0^2}{2vc\rho_0\tilde\rho_0\Gamma_B}$$

$$\delta \equiv \frac{\Omega^2 - \Omega_B^2}{\Gamma_B\Omega}$$

$$k_B = \Omega/v \quad (v = c_{sound})$$

## Normalization substitution

Let $E_a = E_0 a$, $E_b = E_0 b$, and $\rho = i\rho_0\tilde\rho_0 f$

**a and b equations after substitution:**

$$\frac{da}{dz} + \frac{n}{c}\frac{da}{dt} - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$-\frac{\partial b}{\partial z} + \frac{n}{c}\frac{\partial b}{\partial t} + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

**f equation after substitution (build-up):**

$$\frac{\partial \rho}{\partial t} + \left(\frac{\Gamma_B}{2} - \frac{i}{2\Omega}(\Omega^2 - \Omega_B^2)\right)\rho = \frac{i\gamma_e\epsilon_0 k_B^2}{4\Omega}E_aE_b^*$$

$$i\rho_0\tilde\rho_0\frac{df}{dt} + \left[\frac{\Gamma_B}{2} - i\left(\frac{\Omega^2-\Omega_B^2}{2\Omega}\right)\right]i\rho_0\tilde\rho_0 f = \frac{i\gamma_e\epsilon_0ck_B E_0^2}{4vc}ab^*$$

$$\frac{2}{\Gamma_B}\frac{df}{dt} + \left[1 - i\left(\frac{\Omega^2-\Omega_B^2}{\Omega\Gamma_B}\right)\right]f = \frac{\gamma_e\epsilon_0ck_B E_0^2}{2vc\rho_0\tilde\rho_0\Gamma_B}ab^*$$

$$\frac{2}{\Gamma_B}\frac{df}{dt} + (1 - i\delta)f = gab^*$$

## Coordinate transformation

$$\zeta = z + \frac{ct}{n}, \qquad \tau = t$$

$$\frac{\partial}{\partial z} = \frac{\partial}{\partial \zeta}, \qquad \frac{\partial}{\partial t} = \frac{\partial}{\partial \tau} + \frac{c}{n}\frac{\partial}{\partial \zeta}$$

## Substituting the coordinate transform into a and b

$$\frac{\partial a}{\partial \zeta} + \frac{n}{c}\left(\frac{\partial a}{\partial \tau} + \frac{c}{n}\frac{\partial a}{\partial \zeta}\right) - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$-\frac{\partial b}{\partial \zeta} + \frac{n}{c}\left(\frac{\partial b}{\partial \tau} + \frac{c}{n}\frac{\partial b}{\partial \zeta}\right) + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

which simplify to:

$$2\frac{\partial a}{\partial \zeta} + \frac{n}{c}\frac{\partial a}{\partial \tau} - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

## Substituting the coordinate transform into f

$$\frac{2}{\Gamma_B}\left(\frac{\partial f}{\partial \tau} + \frac{c}{n}\frac{\partial f}{\partial \zeta}\right) + (1-i\delta)f = gab^*$$

## Final boxed results

$$2\frac{\partial a}{\partial \zeta} + \frac{n}{c}\frac{\partial a}{\partial \tau} - \frac{i}{2k_a}\nabla_T^2 a = -k_{norm}bf$$

$$\frac{n}{c}\frac{\partial b}{\partial \tau} + \frac{i}{2k_b}\nabla_T^2 b = k_{norm}af^*$$

$$\frac{2}{\Gamma_B}\left(\frac{\partial f}{\partial \tau} + \frac{c}{n}\frac{\partial f}{\partial \zeta}\right) + (1-i\delta)f = gab^*$$
