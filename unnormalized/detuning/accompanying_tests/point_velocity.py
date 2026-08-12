#Run different detuning factors-----------------------------------------------------------------------------

detuning_factors = [0, d/5, 2*d/5, 3*d/5,4*d/5, d]
nruns = len(detuning_factors)
a_runs, b_runs, f_runs = [], [], []
for i, dval in enumerate(detuning_factors):
    a_i, b_i, f_i, *_ = detune(d=dval)
    a_runs.append(a_i); b_runs.append(b_i); f_runs.append(f_i)
    globals()[f"a{i}"], globals()[f"b{i}"], globals()[f"f{i}"] = a_i, b_i, f_i

#Velocities of different points over time for different detuning factors --------------------------------------------
nrows = len(datasets)
fig, axes = plt.subplots(nrows = nrows, ncols = 1, sharex = True, sharey = True, figsize = (10,3.2*nrows))

colors = plt.cm.viridis(np.linspace(0,1, 10))

labels = [10,20,30,40,50,60,70,80,90,100]
for row, (bb,aa,ff,delta) in enumerate(datasets):
    ax = axes[row]
    edge_velocity = point_velocity(b=bb)
    for n in range(10):
        ax.plot(tau*1e9,edge_velocity[:,n], color=colors[n], label = f"{labels[n]}% peak")
    ax.set_ylabel(r"$\zeta$ (m)")
    ax.set_title(rf"$\delta$ = {delta}", loc = "left")

axes[-1].set_xlabel(r"$\tau$ (ns)")
axes[0].legend()
fig.suptitle("Point Velocities for Diffeent Detuning")
fig.tight_layout()
