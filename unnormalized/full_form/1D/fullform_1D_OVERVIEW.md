# Construction of the Full-Form Solver in 1D

Let us briefly focus on the construction of the full-form solver. In 1D, we have

$$\frac{\partial a}{\partial \tau} = -\frac{2c}{n}\frac{\partial a}{\partial \zeta} - \frac{c}{n}k_{norm}bf$$

$$\frac{\partial b}{\partial \tau} = \frac{c}{n}k_{norm}af^{\ast}$$

$$\frac{\partial f}{\partial \tau} = \frac{\Gamma_B}{2}gab^{\ast} - \frac{c}{n}\frac{\partial f}{\partial \zeta} - \frac{\Gamma_B}{2}(1-i\delta)f$$

Here, different parts evolve on very different timescales. We will solve using IMEX RK methods.

# IMEX Runge–Kutta

**IMEX**: **im**plicit and **ex**plicit, i.e. distinct handling of the implicit and explicit parts, with RK-style stage weightings.

Generally, an RK4 tableau reveals the stage weightings:

$$\begin{matrix} 0 & \vert &  &  &  &  \cr \frac{1}{2} & \vert & \frac{1}{2} &  &  &  \cr \frac{1}{2} & \vert & 0 & \frac{1}{2} &  &  \cr 1 & \vert & 0 & 0 & 1 &  \cr  & \vert & \overline{\,\frac{1}{6}\,} & \overline{\,\frac{1}{3}\,} & \overline{\,\frac{1}{3}\,} & \overline{\,\frac{1}{6}\,} \end{matrix} = \begin{matrix} \vec{c} & \vert & A \cr  & \vert & \overline{\,\vec{w}^T\,} \end{matrix} \qquad \frac{dy}{dt} = f(t,y)$$

The update and the stages are

$$Y_{t+\Delta t} = Y_t + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) = Y_t + \Delta t\sum_j w_j k_j$$

$$k_j = f\left(t + c_j\Delta t, \ Y_t + \Delta t\sum_i A_{ji}k_i\right)$$

That is, the ith row of $\vec{c}$ gives the fraction $[0,1]$ of the step at which the ith stage approximation is evaluated. $A_{ji}$ gives the approximation contributions from previous stages. Finally, $w_j$ gives the weight of each stage in the final value.

For explicit methods, $[A]$ is strictly lower-triangular, so $k_j$ does not rely on $k_j$ (nor on any later stage). Also,

$$1 = \sum_j w_j$$

## Splitting the Full-Form Equations

For the full-form equations, we can divide our system into

$$\frac{\partial \vec{U}}{\partial \tau} = L(\vec{U}) + N(\vec{U})$$

where $L$ is the linear, stiff part and $N$ is the non-linear, non-stiff part, and

$$\vec{U}(\tau) = \begin{pmatrix} a_{\zeta_0}(\tau) \cr a_{\zeta_1}(\tau) \cr \vdots \cr a_{\zeta_f}(\tau) \cr b_{\zeta_0}(\tau) \cr \vdots \cr b_{\zeta_f}(\tau) \cr f_{\zeta_0}(\tau) \cr \vdots \cr f_{\zeta_f}(\tau) \end{pmatrix}$$

[Vectorizing the full set allows inter-field implicit calculations.]

We now have:

**$L$** [everything is linear, so this is a linear map/matrix]:

$$L: \begin{cases} \dfrac{\partial a}{\partial \tau} = -\dfrac{2c}{n}\dfrac{\partial a}{\partial \zeta} \cr \dfrac{\partial b}{\partial \tau} = 0 \cr \dfrac{\partial f}{\partial \tau} = -\dfrac{c}{n}\dfrac{\partial f}{\partial \zeta} - \dfrac{\Gamma_B}{2}(1-i\delta)f \end{cases}$$

**$N$** [not linear; will require term-by-term manipulation]:

$$N: \begin{cases} \dfrac{\partial a}{\partial \tau} = -\dfrac{c}{n}k_{norm}bf \cr \dfrac{\partial b}{\partial \tau} = \dfrac{c}{n}k_{norm}af^{\ast} \cr \dfrac{\partial f}{\partial \tau} = \dfrac{\Gamma_B}{2}gab^{\ast} \end{cases}$$

## Tableaus (Corrected)

With the ARS(2,2,2) constants

$$\theta = 1 - \frac{1}{\sqrt{2}}, \qquad \eta = 1 - \frac{1}{2\theta} = -\frac{1}{\sqrt{2}}$$

**For $L$** ($\vec{c}_L$, $A_L$, $\vec{w}_L$):

$$\begin{matrix} 0 & \vert & 0 & 0 & 0 \cr \theta & \vert & 0 & \theta & 0 \cr 1 & \vert & 0 & 1-\theta & \theta \cr  & \vert & \overline{\,0\,} & \overline{\,1-\theta\,} & \overline{\,\theta\,} \end{matrix}$$

**For $N$** ($\vec{c}_N$, $A_N$, $\vec{w}_N$):

$$\begin{matrix} 0 & \vert & 0 & 0 & 0 \cr \theta & \vert & \theta & 0 & 0 \cr 1 & \vert & \eta & 1-\eta & 0 \cr  & \vert & \overline{\,\eta\,} & \overline{\,1-\eta\,} & \overline{\,0\,} \end{matrix}$$

[Rows 1, 2, 3 of each tableau give the contributions for $\vec{V}^0$, $\vec{V}^1$, $\vec{V}^2$, respectively.]

Because the weights equal the last row of $A$ in each tableau,

$$\vec{V}(\tau + \Delta\tau) = \vec{V}^2(\tau)$$

## Skeleton Algorithm

**Step 1.** Solve $N(\vec{V}^0(\tau))$. [enforce BCs here...]

**Step 2.** Implicit solve for $\vec{V}^1(\tau)$:

$$\vec{V}^1(\tau) = \vec{V}^0(\tau) + \Delta\tau\Big(\theta L(\vec{V}^1(\tau)) + \theta N(\vec{V}^0(\tau))\Big)$$

$$\Longrightarrow \quad (I - \Delta\tau\theta L)\vec{V}^1(\tau) = \vec{V}^0(\tau) + \Delta\tau\theta N(\vec{V}^0(\tau))$$

**Step 3.** Solve for $N(\vec{V}^1(\tau))$. [and here.]

**Step 4.** Implicit solve for $\vec{V}^2(\tau)$:

$$\vec{V}^2(\tau) = \vec{V}^0(\tau) + \Delta\tau\Big(\theta L(\vec{V}^2(\tau)) + (1-\theta)L(\vec{V}^1(\tau)) + \eta N(\vec{V}^0(\tau)) + (1-\eta)N(\vec{V}^1(\tau))\Big)$$

$$\Longrightarrow \quad [I - \Delta\tau\theta L]\vec{V}^2(\tau) = \vec{V}^0(\tau) + \Delta\tau\Big[(1-\theta)L(\vec{V}^1(\tau)) + \eta N(\vec{V}^0(\tau)) + (1-\eta)N(\vec{V}^1(\tau))\Big]$$

**Step 5.** $\vec{V}^0(\tau + \Delta\tau) = \vec{V}^2(\tau)$. Repeat for all $\tau$.

# The Nonlinear Operator $N$

$N(\vec{V})$, where

$$\vec{V} = \begin{pmatrix} \vec{a} \cr \vec{b} \cr \vec{f} \end{pmatrix}$$

$$N(\vec{V}) = \begin{pmatrix} -\frac{c}{n}k_{norm}b_{\zeta_0}f_{\zeta_0} \cr \vdots \cr -\frac{c}{n}k_{norm}b_{\zeta_f}f_{\zeta_f} \cr \frac{c}{n}k_{norm}a_{\zeta_0}f^{\ast}_{\zeta_0} \cr \vdots \cr \frac{c}{n}k_{norm}a_{\zeta_f}f^{\ast}_{\zeta_f} \cr \frac{\Gamma_B}{2}ga_{\zeta_0}b^{\ast}_{\zeta_0} \cr \vdots \cr \frac{\Gamma_B}{2}ga_{\zeta_f}b^{\ast}_{\zeta_f} \end{pmatrix} = \begin{pmatrix} \partial a_{\zeta_0}/\partial\tau \cr \vdots \cr \partial a_{\zeta_f}/\partial\tau \cr \partial b_{\zeta_0}/\partial\tau \cr \vdots \cr \partial b_{\zeta_f}/\partial\tau \cr \partial f_{\zeta_0}/\partial\tau \cr \vdots \cr \partial f_{\zeta_f}/\partial\tau \end{pmatrix}$$

[The right-hand column is the nonlinear contribution to each time derivative.]

Couple the sub-vectors (products taken component by component, $\odot$):

$$\frac{\partial \vec{a}}{\partial \tau} = -\frac{ck_{norm}}{n}\vec{b}\odot\vec{f}$$

$$\frac{\partial \vec{b}}{\partial \tau} = \frac{ck_{norm}}{n}\vec{a}\odot\vec{f}^{\ast}$$

$$\frac{\partial \vec{f}}{\partial \tau} = \frac{\Gamma_B}{2}g\vec{a}\odot\vec{b}^{\ast}$$

# The Linear Operator $L$

For $L(\vec{V})$, generally (centered differences):

$$\frac{\partial a}{\partial \tau} = -\frac{2c}{n}\frac{\partial a}{\partial \zeta} = -\frac{2c}{n}\left(\frac{a_{\zeta+d\zeta} - a_{\zeta-d\zeta}}{2d\zeta}\right)$$

$$\frac{\partial b}{\partial \tau} = 0$$

$$\frac{\partial f}{\partial \tau} = -\frac{c}{n}\frac{\partial f}{\partial \zeta} - \frac{\Gamma_B}{2}(1-i\delta)f = -\frac{c}{n}\left(\frac{f_{\zeta+d\zeta} - f_{\zeta-d\zeta}}{2d\zeta}\right) - \frac{\Gamma_B}{2}(1-i\delta)f_\zeta$$

Or:

$$\frac{\partial a_\zeta}{\partial \tau} = L(a) = \underbrace{\frac{2c}{n(2d\zeta)}}_{{}_aL_1}a_{\zeta-d\zeta} + \underbrace{0}_{{}_aL_2}a_\zeta + \underbrace{\left(-\frac{2c}{n(2d\zeta)}\right)}_{{}_aL_3}a_{\zeta+d\zeta}$$

$$\frac{\partial b_\zeta}{\partial \tau} = 0 \text{ everywhere}, \qquad L(b) = 0$$

$$\frac{\partial f_\zeta}{\partial \tau} = L(f) = \underbrace{\frac{c}{n(2d\zeta)}}_{{}_fL_1}f_{\zeta-d\zeta} + \underbrace{\left(-\frac{\Gamma_B}{2}(1-i\delta)\right)}_{{}_fL_2}f_\zeta + \underbrace{\left(-\frac{c}{n(2d\zeta)}\right)}_{{}_fL_3}f_{\zeta+d\zeta}$$

$L$ is block diagonal, with a tridiagonal block for each field (${}_aL_1, {}_bL_1, {}_fL_1$ on the sub-diagonal; ${}_aL_2, {}_bL_2, {}_fL_2$ on the diagonal; ${}_aL_3, {}_bL_3, {}_fL_3$ on the super-diagonal; ${}_bL_1 = {}_bL_2 = {}_bL_3 = 0$):

$$L = \begin{pmatrix} L_a & 0 & 0 \cr 0 & L_b & 0 \cr 0 & 0 & L_f \end{pmatrix}$$

$$L_a = \begin{pmatrix} {}_aL_2 & {}_aL_3 & & \cr {}_aL_1 & {}_aL_2 & {}_aL_3 & \cr & \ddots & \ddots & \ddots \cr & & {}_aL_1 & {}_aL_2 \end{pmatrix}$$

$$L_b = \begin{pmatrix} {}_bL_2 & {}_bL_3 & & \cr {}_bL_1 & {}_bL_2 & {}_bL_3 & \cr & \ddots & \ddots & \ddots \cr & & {}_bL_1 & {}_bL_2 \end{pmatrix} = 0$$

$$L_f = \begin{pmatrix} {}_fL_2 & {}_fL_3 & & \cr {}_fL_1 & {}_fL_2 & {}_fL_3 & \cr & \ddots & \ddots & \ddots \cr & & {}_fL_1 & {}_fL_2 \end{pmatrix}$$

# Boundary Conditions

**At the $+\infty$ boundary**, take (backward difference)

$$\frac{\partial a}{\partial \tau} = -\frac{2c}{n}\left(\frac{a_\zeta - a_{\zeta-d\zeta}}{d\zeta}\right) = \frac{2c}{nd\zeta}a_{\zeta-d\zeta} - \frac{2c}{nd\zeta}a_\zeta$$

$$\frac{\partial f}{\partial \tau} = -\frac{c}{n}\left(\frac{f_\zeta - f_{\zeta-d\zeta}}{d\zeta}\right) - \frac{\Gamma_B}{2}(1-i\delta)f_\zeta = \frac{c}{nd\zeta}f_{\zeta-d\zeta} - \left[\frac{c}{nd\zeta} + \frac{\Gamma_B}{2}(1-i\delta)\right]f_\zeta$$

[These replace the last row of the $a$ and $f$ blocks of $L$.]

**At the $-\infty$ boundary**, we must enforce the BCs: undepleted pump, no acoustic. Furthermore, we can take the following: $(a,b,f)$ at $\zeta \to -\infty$ do not evolve by the couplings, because the effects generally flow towards $+\zeta$. Hence, we can set the diagonal entries to zero,

$${}_aL_{2,\zeta_0} = {}_bL_{2,\zeta_0} = {}_fL_{2,\zeta_0} = 0$$

[so the diagonal of $I - \Delta\tau\theta L$ becomes 1 in the operations], and likewise the off-diagonal entries in the first row of each block, so that each boundary row of $I - \Delta\tau\theta L$ is a pure identity row:

$${}_aL_{1,\zeta_0} = {}_aL_{3,\zeta_0} = 0, \qquad {}_fL_{1,\zeta_0} = {}_fL_{3,\zeta_0} = 0$$

while enforcing the boundary values (on the right-hand side of the implicit solves)

$$V(a_{\zeta_0}) = a_0, \qquad V(f_{\zeta_0}) = 0$$
