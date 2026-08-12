## Real and imaginary parts (finite damping)

With $\Gamma_B$ finite the acoustic field is no longer slaved to the optical beating.

Instead we split each field into Cartesian real and imaginary parts:

$$a = a_R + ia_I, \qquad b = b_R + ib_I, \qquad f = f_R + if_I$$

All six components $a_R, a_I, b_R, b_I, f_R, f_I$ are real, and every coefficient ($k_{norm}$, $g$, $n/c$, $\Gamma_B$) is real.

### Pump

$$2\frac{\partial a}{\partial \zeta} = -k_{norm}bf$$

Expand the product:

$$bf = (b_R f_R - b_I f_I) + i(b_R f_I + b_I f_R)$$

$$2\frac{\partial a_R}{\partial \zeta} + 2i\frac{\partial a_I}{\partial \zeta} = -k_{norm}\big[(b_R f_R - b_I f_I) + i(b_R f_I + b_I f_R)\big]$$

Matching real and imaginary parts:

$$2\frac{\partial a_R}{\partial \zeta} = -k_{norm}(b_R f_R - b_I f_I)$$

$$2\frac{\partial a_I}{\partial \zeta} = -k_{norm}(b_R f_I + b_I f_R)$$

### Seed

$$\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}af^*$$

Here $f^* = f_R - if_I$, so

$$af^* = (a_R f_R + a_I f_I) + i(a_I f_R - a_R f_I)$$

giving

$$\frac{n}{c}\frac{\partial b_R}{\partial \tau} + i\frac{n}{c}\frac{\partial b_I}{\partial \tau} = k_{norm}\big[(a_R f_R + a_I f_I) + i(a_I f_R - a_R f_I)\big]$$

Matching parts:

$$\frac{n}{c}\frac{\partial b_R}{\partial \tau} = k_{norm}(a_R f_R + a_I f_I)$$

$$\frac{n}{c}\frac{\partial b_I}{\partial \tau} = k_{norm}(a_I f_R - a_R f_I)$$

### Acoustic

$$\frac{2c}{n\Gamma_B}\frac{\partial f}{\partial \zeta} + f = gab^*$$

Here $b^* = b_R - ib_I$, so

$$ab^* = (a_R b_R + a_I b_I) + i(a_I b_R - a_R b_I)$$

giving

$$\frac{2c}{n\Gamma_B}\left(\frac{\partial f_R}{\partial \zeta} + i\frac{\partial f_I}{\partial \zeta}\right) + (f_R + if_I) = g\big[(a_R b_R + a_I b_I) + i(a_I b_R - a_R b_I)\big]$$

Matching parts:

$$\frac{2c}{n\Gamma_B}\frac{\partial f_R}{\partial \zeta} + f_R = g(a_R b_R + a_I b_I)$$

$$\frac{2c}{n\Gamma_B}\frac{\partial f_I}{\partial \zeta} + f_I = g(a_I b_R - a_R b_I)$$

### Result

The six equations remain fully coupled.

Unlike the strong-damping limit, the imaginary parts $a_I, b_I, f_I$ each carry their own source terms and cannot be set to zero without loss of generality.

Physically, $f$ now has independent longitudinal dynamics, so its phase is no longer pinned to $\phi_a - \phi_b$ — the automatic phase-matching that made the envelope phase redundant in the $\Gamma_B \to \infty$ case is broken here.
