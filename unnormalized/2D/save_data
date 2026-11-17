#Save
np.savez('TEST.npz', 
         physical_params = {"G_b": G_b, "f_B": f_B, "rho_0": rho_0, "n": n, "gamma": gamma, "dlambda": 0},
         run_params = {"a0": a0, "p_sigma": p_sigma, "b0": b0, "sigma": sigma, "w0": w0, "F": F, "focus_dist": time_to_focus*c_light/n},
         a=a, b=b, f=f,
         rad=rad, zeta=zeta, tau=tau,
         phi_a = phi_a, phi_b = phi_b, phi_f = phi_f, b_axis= b_axis,
         a_in_col = a_in_col, a_out_col = a_out_col)

#load
filedata = np.load("rundata/square_intentions/t1.npz")
a = filedata["a"]
b = filedata["b"]
f = filedata["f"]
b_axis = filedata["b_axis"]
a_in_col = filedata["a_in_col"]
a_out_col = filedata["a_out_col"]

"""
Saving is necessary for 2D because of long runtimes"""
