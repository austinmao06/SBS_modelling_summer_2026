This folder uses the equations based on the normalized coordinate transformations. These integrators were used to verify against the previous work done within the group. They are a good starting point, but they get a bit ugly when trying to work with physical units.


**The transformations go as follows (Note $\Omega = \Omega_B$):**

## Starting equations

$$\frac{\partial E_a}{\partial z} + \frac{n}{c}\frac{\partial E_a}{\partial t} = \frac{i\omega_a \gamma_e}{4cn\rho_0} E_b \rho$$

$$-\frac{\partial E_b}{\partial z} + \frac{n}{c}\frac{\partial E_b}{\partial t} = \frac{i\omega_b \gamma_e}{4cn\rho_0} E_a \rho^*$$

$$\frac{\partial \rho}{\partial t} + \frac{\Gamma_B}{2}\rho = \frac{i\gamma_e \epsilon_0 K_B}{4v} E_a E_b^*$$

## Definitions

$$a = \frac{E_a}{E_0}, \quad b = \frac{E_b}{E_0}, \quad f = \frac{\rho}{i\bar\rho_0 \rho_0}$$

$$\tilde z = k_{Norm} z, \quad \tilde t = \omega_{Norm}t, \quad \omega = \omega_a = \omega_b$$

$$\omega_{Norm} = \frac{c k_{Norm}}{n}, \quad k_{Norm} = \frac{\bar\rho_0 \gamma_e \omega}{4nc}$$

## Step 1 -- Normalize a

Substitute $E_a = a E_0$ and the scaled derivatives $\partial_z = k_{Norm}\partial_{\tilde z}$, $\partial_t = \omega_{Norm}\partial_{\tilde t}$:

$$k_{Norm}\frac{da}{d\tilde z} + \frac{\omega_{Norm} n}{c}\frac{da}{d\tilde t} = \frac{i\rho}{\rho_0}\cdot\frac{\gamma_e \omega_a}{4nc}b$$

Substitute $\rho = i\bar\rho_0\rho_0 f$:

$$k_{Norm}\frac{da}{d\tilde z} + \frac{\omega_{Norm} n}{c}\frac{da}{d\tilde t} = -\frac{\bar\rho_0\gamma_e\omega_a}{4nc}bf$$

Divide through by $k_{Norm}$, using $\omega_{Norm} n / c = k_{Norm}$:

$$\frac{da}{d\tilde z} + \frac{da}{d\tilde t} = -bf$$

## Step 2 -- Normalize b

$$-k_{Norm}\frac{db}{d\tilde z} + \frac{\omega_{Norm} n}{c}\frac{db}{d\tilde t} = \frac{\bar\rho_0\gamma_e\omega_b}{4nc}af^*$$

Divide by $k_{Norm}$:

$$-\frac{db}{d\tilde z} + \frac{db}{d\tilde t} = af^*$$

## Step 3 -- Normalize f

$$i\bar\rho_0\rho_0\omega_{Norm}\frac{df}{d\tilde t} + i\bar\rho_0\rho_0\frac{\Gamma_B}{2}f = \frac{i\gamma_e\epsilon_0 K_B}{4v}ab^* E_0^2$$

Divide by $i\bar\rho_0\rho_0$:

$$\omega_{Norm}\frac{df}{d\tilde t} + \frac{\Gamma_B}{2}f = \frac{\gamma_e\epsilon_0 K_B E_0^2}{4v\bar\rho_0\rho_0} ab^*$$

Multiply through by $2/\Gamma_B$:

$$\frac{2\omega_{Norm}}{\Gamma_B}\frac{df}{d\tilde t} + f = \frac{\gamma_e\epsilon_0 K_B E_0^2}{2v\Gamma_B\bar\rho_0\rho_0} ab^* = gab^*$$

where

$$g = \frac{\gamma_e\epsilon_0 K_B E_0^2}{2v\Gamma_B\bar\rho_0\rho_0}, \qquad \Gamma = \frac{\Gamma_B}{2\omega_{Norm}}$$

$$f + \frac{1}{\Gamma}\frac{df}{d\tilde t} = gab^*$$

## Summary -- the three normalized equations

$$\frac{da}{d\tilde z} + \frac{da}{d\tilde t} = -bf$$

$$-\frac{db}{d\tilde z} + \frac{db}{d\tilde t} = af^*$$

$$f + \frac{1}{\Gamma}\frac{df}{d\tilde t} = gab^*$$

## Coordinate transformation to Lab Frame

$$\zeta = \frac{\tilde z + \tilde t}{2}, \qquad \tau = \tilde t$$

$$\frac{\partial}{\partial \tilde z} = \frac{1}{2}\frac{\partial}{\partial \zeta}, \qquad \frac{\partial}{\partial \tilde t} = \frac{1}{2}\frac{\partial}{\partial \zeta} + \frac{\partial}{\partial \tau}$$
## Setting up the chain rule

We have $\zeta = \dfrac{\tilde t + \tilde z}{2}$, $\tau = \tilde t$, so inverting: $\tilde z = 2\zeta - \tau$, $\tilde t = \tau$.

For any function $F(\tilde z, \tilde t)$, expressed as $F(\zeta, \tau)$:

$$\frac{\partial F}{\partial \tilde z} = \frac{\partial F}{\partial \zeta}\frac{\partial \zeta}{\partial \tilde z} + \frac{\partial F}{\partial \tau}\frac{\partial \tau}{\partial \tilde z}$$

$$\frac{\partial F}{\partial \tilde t} = \frac{\partial F}{\partial \zeta}\frac{\partial \zeta}{\partial \tilde t} + \frac{\partial F}{\partial \tau}\frac{\partial \tau}{\partial \tilde t}$$

With $\dfrac{\partial \zeta}{\partial \tilde z} = \dfrac{1}{2}$, $\dfrac{\partial \zeta}{\partial \tilde t} = \dfrac{1}{2}$, $\dfrac{\partial \tau}{\partial \tilde z} = 0$, $\dfrac{\partial \tau}{\partial \tilde t} = 1$:

$$\frac{\partial F}{\partial \tilde z} = \frac{1}{2}\frac{\partial F}{\partial \zeta}, \qquad \frac{\partial F}{\partial \tilde t} = \frac{1}{2}\frac{\partial F}{\partial \zeta} + \frac{\partial F}{\partial \tau}$$

This holds for $F = a$, $F = b$, and $F = f$ alike.

## Substituting into all three equations

**a equation:**

$$\frac{\partial a}{\partial \tilde z} + \frac{\partial a}{\partial \tilde t} = \frac{1}{2}\frac{\partial a}{\partial \zeta} + \left(\frac{1}{2}\frac{\partial a}{\partial \zeta} + \frac{\partial a}{\partial \tau}\right) = \frac{\partial a}{\partial \zeta} + \frac{\partial a}{\partial \tau}$$

$$\frac{\partial a}{\partial \zeta} + \frac{\partial a}{\partial \tau} = -bf$$

**b equation:**

$$-\frac{\partial b}{\partial \tilde z} + \frac{\partial b}{\partial \tilde t} = -\frac{1}{2}\frac{\partial b}{\partial \zeta} + \left(\frac{1}{2}\frac{\partial b}{\partial \zeta} + \frac{\partial b}{\partial \tau}\right) = \frac{\partial b}{\partial \tau}$$

$$\frac{\partial b}{\partial \tau} = af^*$$

**f equation:**

$$f + \frac{1}{\Gamma}\left(\frac{1}{2}\frac{\partial f}{\partial \zeta} + \frac{\partial f}{\partial \tau}\right) = gab^*$$

$$f + \frac{1}{2\Gamma}\frac{\partial f}{\partial \zeta} + \frac{1}{\Gamma}\frac{\partial f}{\partial \tau} = gab^*$$


For $a$, assume $\dfrac{\partial a}{\partial \zeta} \gg \dfrac{\partial a}{\partial \tau}$, so the $\tau$-term is dropped:

$$\boxed{\frac{\partial a}{\partial \zeta} = -bf}$$

For $b$, there is no $\zeta$-derivative in the equation to begin with, so the approximation doesn't apply — it is unchanged:

$$\boxed{\frac{\partial b}{\partial \tau} = af^*}$$

For $f$, assume $\dfrac{\partial f}{\partial \zeta} \gg \dfrac{\partial f}{\partial \tau}$, so the $\tau$-term is dropped:

$$\boxed{f + \frac{1}{2\Gamma}\frac{\partial f}{\partial \zeta} = gab^*}$$

## Final reduced equations

$$\frac{\partial a}{\partial \zeta} = -bf$$

$$\frac{\partial b}{\partial \tau} = af^*$$

$$f + \frac{1}{2\Gamma}\frac{\partial f}{\partial \zeta} = gab^*$$
