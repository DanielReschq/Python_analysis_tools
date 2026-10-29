import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import Collected_runs as cr
import FS3
import sympy
import glob
import time
import scipy as sp
import parameters

import warnings




#######

plt.close("all")

#cl = cr.collected_runs("/home/daniel/Master_thesis/Data/LocalData/runs_Ts000_1sweep",False,0)
cl = cr.collected_runs("/home/daniel/Master_thesis/Data/Leo4Data/runs_Js000",False,0)


for L in cl.L_list:
    run =cl.L_dict[L][5]
    data = run.data

    plt.plot(data["mag4"])
plt.xlim(0,200)
plt.show()

data = cl.L_dict[128][0].data

for obs in cl.obs_names:
    
    dic = binning_analysis(data[obs],min_bins=32,n_plateau=4,n_sigma=1.0)
    print(obs,dic["converged"],dic["tau_int"])


    plt.errorbar(dic["bin_sizes"],dic["errors"]/dic["errors"][-1],yerr=dic["errors_err"]/dic["errors"][-1],label=cl.obs_labels[obs],marker="o")

plt.xlabel(r"$n_{sweeps}$")
plt.ylabel(r"$\sigma$(%s)"%cl.obs_labels[obs])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

# 