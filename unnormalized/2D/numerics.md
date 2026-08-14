# Numerical Treatment of 2D Equations

The inclusion of the diffraction operator makes the system less straightforward to numerically solve. The formulation present in the code employs the method of Strang Splitting.
Strang Splitting replaces the blocks of Forward Euler, backward Euler, and RK4. Strang Splitting allows distinct treatment of the diffraction and field-coupling operators while maintaining the same advance method of $\zeta$ for a, f and $\tau$ for b.
The blocks are replaced by the following Strang-Split blocks:

--half-step Crank-Nicolson for diffraction
--full-step Euler/RK4 for field coupling
--half-step Crank-Nicolson for diffraction

# Crank–Nicolson Formulation (Axisymmetric Diffraction)

Crank–Nicolson (CN) treatment of the transverse (radial) diffraction operator, applied separately to the pump $a$ and seed $b$.

Axial symmetry is assumed ($\partial/\partial\theta = 0$), so only radial derivatives survive from the transverse Laplacian.

Only the diffraction operator $D$ is stepped here; the reaction/source term is handled in a separate substep of the splitting.

The pump marches in $\zeta$; the seed marches in $\tau$.


## Pump ($a$)

**Full equation**

$$\frac{\partial a}{\partial \zeta} = \frac{i}{4k_L}\left(\frac{\partial^2 a}{\partial r^2} + \frac{1}{r}\frac{\partial a}{\partial r}\right) - \frac{k_{norm}}{2}bf$$

**Taylor Expansion Discretization of Diffraction Operator**

$$a(r+dr) = a(r) + \frac{\partial a}{\partial r}dr + \frac{\partial^2 a}{\partial r^2}\frac{dr^2}{2} + O(dr^3)$$

$$a(r-dr) = a(r) - \frac{\partial a}{\partial r}dr + \frac{\partial^2 a}{\partial r^2}\frac{dr^2}{2} + O(dr^3)$$

For second derivative:

$$\frac{\partial^2 a}{\partial r^2} = \frac{a_{r+dr} + a_{r-dr} - 2a_r}{dr^2}$$

For first derivative:

$$\frac{1}{r}\frac{\partial a}{\partial r} = \frac{a_{r+dr} - a_{r-dr}}{2r dr}$$

**Diffraction operator $D$**

$$D \equiv \frac{\partial a}{\partial \zeta} = \frac{i}{4k_L}\left[\left(\frac{1}{dr^2} + \frac{1}{2r dr}\right)a_{r+dr} + \left(\frac{1}{dr^2} - \frac{1}{2r dr}\right)a_{r-dr} - \frac{2}{dr^2}a_r\right]$$

**Crank–Nicolson half-step.** Average $D$ over the old ($\zeta$) and mid levels:

$$\frac{a_r^{mid} - a_r^{\zeta}}{d\zeta/2} = \frac{1}{2}\left[D(a^{mid}) + D(a^{\zeta})\right]$$

Expanded:

$$a_r^{mid} - a_r^{\zeta} = \frac{i d\zeta}{16k_L}\left[\frac{2r+dr}{2r dr^2}(a_{r+dr}^{mid} + a_{r+dr}^{\zeta}) + \frac{2r-dr}{2r dr^2}(a_{r-dr}^{mid} + a_{r-dr}^{\zeta}) - \frac{2}{dr^2}(a_r^{mid} + a_r^{\zeta})\right]$$

**Linear equation for a general (interior) $r$.**

$$a_{r-dr}^{mid}(-{}_aK_1) + a_r^{mid}(1 + {}_aK_2) + a_{r+dr}^{mid}(-{}_aK_3) = a_{r-dr}^{\zeta}({}_aK_1) + a_r^{\zeta}(1 - {}_aK_2) + a_{r+dr}^{\zeta}({}_aK_3)$$

$${}_aK_1 = \frac{i d\zeta (2r - dr)}{32 r k_L dr^2}$$

$${}_aK_2 = \frac{i d\zeta}{8 k_L dr^2}$$

$${}_aK_3 = \frac{i d\zeta (2r + dr)}{32 r k_L dr^2}$$

**Matrix form.** Unknown is the mid vector; the right side is built from the known $\zeta$ level. Each coefficient depends on $r$, so it differs row to row. Rows shown are interior nodes ($r_0$ and $r_{Max}$ get separate boundary treatment).

$$\begin{pmatrix} \ddots & \ddots & & \\ -{}_aK_1 & 1+{}_aK_2 & -{}_aK_3 & \\ & -{}_aK_1 & 1+{}_aK_2 & -{}_aK_3 \\ & & \ddots & \ddots \end{pmatrix} \begin{pmatrix} \vdots \\ a_{r_n}^{mid} \\ a_{r_{n+1}}^{mid} \\ \vdots \end{pmatrix} = \begin{pmatrix} \vdots \\ {}_aK_1 a_{r_{n-1}}^{\zeta} + (1-{}_aK_2) a_{r_n}^{\zeta} + {}_aK_3 a_{r_{n+1}}^{\zeta} \\ {}_aK_1 a_{r_n}^{\zeta} + (1-{}_aK_2) a_{r_{n+1}}^{\zeta} + {}_aK_3 a_{r_{n+2}}^{\zeta} \\ \vdots \end{pmatrix}$$

The tridiagonal system can be solved directly with NumPy.


## Seed ($b$)

**Full equation**

$$\frac{\partial b}{\partial \tau} = -\frac{ic}{2k_L n}\left(\frac{\partial^2 b}{\partial r^2} + \frac{1}{r}\frac{\partial b}{\partial r}\right) + \frac{c k_{norm}}{n}af^*$$

**Taylor Expansion Discretization of Diffraction Operator**

$$b(r+dr) = b(r) + \frac{\partial b}{\partial r}dr + \frac{\partial^2 b}{\partial r^2}\frac{dr^2}{2} + O(dr^3)$$

$$b(r-dr) = b(r) - \frac{\partial b}{\partial r}dr + \frac{\partial^2 b}{\partial r^2}\frac{dr^2}{2} + O(dr^3)$$

For second derivative:

$$\frac{\partial^2 b}{\partial r^2} = \frac{b_{r+dr} + b_{r-dr} - 2b_r}{dr^2}$$

For first derivative:

$$\frac{1}{r}\frac{\partial b}{\partial r} = \frac{b_{r+dr} - b_{r-dr}}{2r dr}$$

**Diffraction operator $D$**

$$D \equiv \frac{\partial b}{\partial \tau} = -\frac{ic}{2k_L n}\left[\left(\frac{1}{dr^2} + \frac{1}{2r dr}\right)b_{r+dr} + \left(\frac{1}{dr^2} - \frac{1}{2r dr}\right)b_{r-dr} - \frac{2}{dr^2}b_r\right]$$

**Crank–Nicolson half-step.** Average $D$ over the old ($\tau$) and mid levels:

$$\frac{b_r^{mid} - b_r^{\tau}}{d\tau/2} = \frac{1}{2}\left[D(b^{mid}) + D(b^{\tau})\right]$$

Expanded:

$$b_r^{mid} - b_r^{\tau} = -\frac{ic d\tau}{8k_L n}\left[\frac{2r+dr}{2r dr^2}(b_{r+dr}^{mid} + b_{r+dr}^{\tau}) + \frac{2r-dr}{2r dr^2}(b_{r-dr}^{mid} + b_{r-dr}^{\tau}) - \frac{2}{dr^2}(b_r^{mid} + b_r^{\tau})\right]$$

**Linear equation for a general (interior) $r$.**

$$b_{r-dr}^{mid}(-{}_bK_1) + b_r^{mid}(1 + {}_bK_2) + b_{r+dr}^{mid}(-{}_bK_3) = b_{r-dr}^{\tau}({}_bK_1) + b_r^{\tau}(1 - {}_bK_2) + b_{r+dr}^{\tau}({}_bK_3)$$

$${}_bK_1 = -\frac{ic d\tau (2r - dr)}{16 r k_L n dr^2}$$

$${}_bK_2 = -\frac{ic d\tau}{4 k_L n dr^2}$$

$${}_bK_3 = -\frac{ic d\tau (2r + dr)}{16 r k_L n dr^2}$$

**Matrix form.** Unknown is the mid vector; the right side is built from the known $\tau$ level. Each coefficient depends on $r$, so it differs row to row. Rows shown are interior nodes ($r_0$ and $r_{Max}$ get separate boundary treatment).

$$\begin{pmatrix} \ddots & \ddots & & \\ -{}_bK_1 & 1+{}_bK_2 & -{}_bK_3 & \\ & -{}_bK_1 & 1+{}_bK_2 & -{}_bK_3 \\ & & \ddots & \ddots \end{pmatrix} \begin{pmatrix} \vdots \\ b_{r_n}^{mid} \\ b_{r_{n+1}}^{mid} \\ \vdots \end{pmatrix} = \begin{pmatrix} \vdots \\ {}_bK_1 b_{r_{n-1}}^{\tau} + (1-{}_bK_2) b_{r_n}^{\tau} + {}_bK_3 b_{r_{n+1}}^{\tau} \\ {}_bK_1 b_{r_n}^{\tau} + (1-{}_bK_2) b_{r_{n+1}}^{\tau} + {}_bK_3 b_{r_{n+2}}^{\tau} \\ \vdots \end{pmatrix}$$

The tridiagonal system is solved directly with NumPy.
