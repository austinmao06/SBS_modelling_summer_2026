## Complex envelopes — the phase is redundant

Write each field as a real amplitude times a phase:

$$a = Ae^{i\phi_a}, \qquad b = Be^{i\phi_b}, \qquad f = Fe^{i\phi_f}$$
with $A,B,F$ and $\phi_a,\phi_b,\phi_f$ all real.

### Acoustic equation — fixes the phase relation

Strong damping gives the algebraic slaving $f = gab^*$. Substituting the envelopes,
$$Fe^{i\phi_f} = g(Ae^{i\phi_a})(Be^{-i\phi_b}) = gABe^{i(\phi_a - \phi_b)}$$

Matching magnitude and phase (both sides are a real amplitude times a unit phasor):
$$\boxed{F = gAB} \qquad\qquad \boxed{\phi_f = \phi_a - \phi_b}$$

So the acoustic phase is not independent — it is locked to the optical phase difference.

### Pump equation

$$2\frac{\partial a}{\partial \zeta} = -k_{norm}bf$$

Chain rule on the LHS splits the derivative into a real and an imaginary operator:

$$\frac{\partial a}{\partial \zeta} = e^{i\phi_a}\Big(\underbrace{\tfrac{\partial A}{\partial \zeta}}_{\text{real}} + \underbrace{iA\tfrac{\partial \phi_a}{\partial \zeta}}_{\text{imag}}\Big)$$

The RHS collects into one phasor, $bf = BFe^{i(\phi_b+\phi_f)}$, so

$$2e^{i\phi_a}\Big(\frac{\partial A}{\partial \zeta} + iA\frac{\partial \phi_a}{\partial \zeta}\Big) = -k_{norm}BFe^{i(\phi_b+\phi_f)}$$

Divide through by $e^{i\phi_a}$ and expand the leftover exponential as $\cos + i\sin$:

$$2\Big(\frac{\partial A}{\partial \zeta} + iA\frac{\partial \phi_a}{\partial \zeta}\Big) = -k_{norm}BF\Big[\cos(\phi_b+\phi_f-\phi_a) + i\sin(\phi_b+\phi_f-\phi_a)\Big]$$

Now impose the acoustic condition $\phi_f = \phi_a - \phi_b$, which makes the argument $\phi_b+\phi_f-\phi_a = 0$. The cosine becomes $1$ and the sine vanishes, so the RHS is purely real. Splitting real and imaginary parts:

$$\text{Re}:\quad 2\frac{\partial A}{\partial \zeta} = -k_{norm}BF \qquad\qquad \text{Im}:\quad \frac{\partial \phi_a}{\partial \zeta} = 0$$

### Seed equation

$$\frac{n}{c}\frac{\partial b}{\partial \tau} = k_{norm}af^*$$

Same steps. LHS: $\frac{\partial b}{\partial \tau} = e^{i\phi_b}\big(\tfrac{\partial B}{\partial \tau} + iB\tfrac{\partial \phi_b}{\partial \tau}\big)$. RHS: $af^* = AFe^{i(\phi_a-\phi_f)}$. Divide by $e^{i\phi_b}$ and expand:

$$\frac{n}{c}\Big(\frac{\partial B}{\partial \tau} + iB\frac{\partial \phi_b}{\partial \tau}\Big) = k_{norm}AF\Big[\cos(\phi_a-\phi_f-\phi_b) + i\sin(\phi_a-\phi_f-\phi_b)\Big]$$

Impose $\phi_f = \phi_a - \phi_b$, so the argument $\phi_a-\phi_f-\phi_b = 0$ again. The RHS is real:

$$\text{Re}:\quad \frac{n}{c}\frac{\partial B}{\partial \tau} = k_{norm}AF \qquad\qquad \text{Im}:\quad \frac{\partial \phi_b}{\partial \tau} = 0$$

### Result

The imaginary parts carry no source term, so the phases never evolve ($\partial_\zeta\phi_a = 0$, $\partial_\tau\phi_b = 0$) and $\phi_f = \phi_a-\phi_b$ stays fixed — they are dynamically inert and may be set to zero. Substituting $F = gAB$ into the real parts recovers the original two-wave system verbatim, with $A,B$ in place of $a,b$:

$$\boxed{2\frac{\partial A}{\partial \zeta} = -k_{norm}gAB^2}$$

$$\boxed{\frac{n}{c}\frac{\partial B}{\partial \tau} = k_{norm}gA^2B}$$
