#Run different detuning factors-----------------------------------------------------------------------------

detuning_factors = [0, d/5, 2*d/5, 3*d/5,4*d/5, d]
nruns = len(detuning_factors)
a_runs, b_runs, f_runs = [], [], []
for i, dval in enumerate(detuning_factors):
    a_i, b_i, f_i, *_ = detune(d=dval)
    a_runs.append(a_i); b_runs.append(b_i); f_runs.append(f_i)
    globals()[f"a{i}"], globals()[f"b{i}"], globals()[f"f{i}"] = a_i, b_i, f_i
#Efficiency over Time-------------------------------------------------------------------------------
time_vals_ns = np.arange(0, 127, 0.5)          # directly specify 0 to 130 ns
time_vals = time_vals_ns / 1e9              # convert ns -> tau (normalized)
times = time_vals_ns                            # already in ns, no need to reconvert

t_idx = np.array([np.argmin(np.abs(tau - t)) for t in time_vals])

# efficiency vs time, one curve per detuning run (auto-adjusts to nruns)
effs_t = [get_eff(a_runs[i])[t_idx] for i in range(nruns)]

colors = plt.cm.viridis(np.linspace(0, 1, nruns))
for i in range(nruns):
    if detuning_factors[i] == 0:
        plt.errorbar(times, effs_t[i], label="No Detuning Reference",
                     color="black", linestyle="--")
    else:
        plt.errorbar(times, effs_t[i], label=f"d = {detuning_factors[i]}", color=colors[i])

plt.xlabel("Time (ns)")
plt.ylabel("Efficiency")
plt.title("Seed Growth Efficiency for Fixed Detuning")
plt.legend()
#plt.ylim(0.9995, 1.0008)
plt.show()
