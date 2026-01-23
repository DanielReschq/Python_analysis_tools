import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sc
import Collected_runs as cr
import FS3
import sympy
import glob
import time

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
        "axes.unicode_minus": True,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.top": "on",
        "xtick.major.bottom": "on",
        "ytick.major.left": "on",
        "ytick.major.right": "on",
        "xtick.top": True,
        "ytick.right": True,
    }
)

parameter_dictT = {
    "path_to_data": "../../dataT/",
    "alpha": [1.5, 2.0, 2.5, 2.75, 3.0, 3.5],
    "sigma": ["_sm05", "_s00", "_s05", "_s075", "_s10", "_s15"],
    # "gRange": [np.array([2.425, 2.435]),np.array([1.339, 1.35]),np.array([0.91, 0.93]),np.array([0.6, 0.65]),np.array([0.2, 0.35])],
    "gRange": [
        np.array([2.425, 2.435]),
        np.array([1.339, 1.35]),
        np.array([0.913, 0.925]),
        np.array([0.77, 0.79]),
        np.array([0.6, 0.67]),
        np.array([0.2, 0.35]),
    ],
    "LRange": [[64, 256], [64, 256], [64, 256], [64, 256], [64, 256], [64, 256]],
    "gc": [2.4334, 1.34227, 0.917, 0.78, 0.60, 0.228463],
    "nu": [1.0, 1.0, 1.4, 1.7, 4.0, 1.75],
    "eta": [1.0, 0.87, 1.0, 1.0, 1.0, 1.0],
    "chiMax": [9.0, 100.9, 800.4, 100.4, 30000, 50000],
    "rho_L_power": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # "JRange": [np.array([4.9,5.2]),np.array([2.89,2.90]),np.array([2.15,2.17]),np.array([0,2.18]),np.array([0,2.18])],
    # "J0": [5.1,2.895,2.165,1.89,2.17],
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}

parameter_dictJ = {
    "path_to_data": "/home/daniel/master_thesis/dataJ/",
    "alpha": [1.5, 2.0, 2.5, 3.0, 3.5, 1e26],
    "sigma": ["_sm05", "_s00", "_s05", "_s10", "_s15", "_SR"],
    "gRange": [
        np.array([4.97, 5.00]),
        np.array([2.89, 2.9]),
        np.array([2.16, 2.17]),
        np.array([1.795, 1.81]),
        np.array([1.59, 1.594]),
        np.array([1.59, 1.594]),
    ],
    "LRange": [[32, 128], [32, 128], [32, 128], [32, 128], [32, 128]],
    "gc": [4.98, 2.897, 2.16, 1.79, 1.592],
    "nu": [0.7, 0.73, 0.69, 0.6, 0.7],
    "eta": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "chiMax": [100000.0, 100000.0, 150000.0, 20000, 8000.0],
    "rho_L_power": [1, 1, 1, 1, 1],
    # "JRange": [np.array([4.9,5.2]),np.array([2.89,2.90]),np.array([2.15,2.17]),np.array([0,2.18]),np.array([0,2.18])],
    # "J0": [5.1,2.895,2.165,1.89,2.17],
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}


def poly(x, *params):
    f = 0
    for n in range(len(params)):
        f += params[n] * x**n
    return f


def shift(f, L, gc, nu, c, omega):
    def g(x, *args):
        return f((x - gc) * L**nu, *args) * (1 + c * L ** (-omega))

    return g


# start_params = 2*np.array(parameter_dictJ["gc"])
# gRange = 2*np.array(parameter_dictJ["gRange"])

# i = 0

# files = glob.glob("/home/daniel/Master_thesis/Data/Leo4DataJ/*")
# files.sort(key=lambda x: cr.collected_runs(x, True, cutoff=0).runs[0].params["alpha"])

# for file in files:
#     path_to_dataT = file

#     cl_runs = cr.collected_runs(path_to_dataT, True, cutoff=0)
#     for run in cl_runs.runs:
#         L = run.params["Lx"]
#         alpha = run.params["alpha"]
#         n = len(run.data["energy"])
#         T = run.params["temp"]
#         print(f"alpha: {alpha}, T: {T}, L: {L}, length: {n}")

#     #cl_runs.select(LRange=[0],gRange=gRange[i])

#     tab1 = cl_runs.get_x_vs_y_data(
#         xname="g_param",
#         yname="mag2",
#         plot=False,
#         yfunc="y/(Ns**2)",
#         ylabel=r"$/N^2$",
#         xscale="linear",
#         yscale="linear",
#     )

#     fitFunc = cr.f_beta_nu()

#     fit_res = cr.data_fit(tab1, fitFunc, polyOrder=3, start_params=[1.3])
#     params, dparams, redChi2, mesg, ierr = fit_res

#     fig = cr.fit_plot(tab1, fitFunc, fit_res, xlabel= r"$U_2$", ylabel=r"$m_s^2$")

#     tab2, fig2, ax2 = cl_runs.get_x_vs_y_data(
#         xname="g_param",
#         yname="mag2",
#         plot=True,
#         yfunc="y*L",
#         ylabel=r"$L$",
#         xscale="linear",
#         yscale="linear",
#     )

#     plt.show()

#     i+=1
# plt.clf()


# path = "/home/daniel/Master_thesis/Data/runs_BETAJ5"

# cl_runs = cr.collected_runs(path, False, cutoff=100)

# cl_runs.get_x_vs_y_data(
#     xname="g_param",
#     yname="spin_stiffness",
#     yfunc="y",
#     xfunc="1/x",
#     plot=True
# )

# plt.show()

path = ["/home/daniel/Master_thesis/Data/Leo4DataJ/runs_Js000",
        "/home/daniel/Master_thesis/Data/runs_Js000"]

cl_runs = cr.collected_runs(path, True, cutoff=5)

# cl_runs.select(LRange=[32, 256], gRange=[1.3, 1.346])

data, _, _ = cl_runs.get_x_vs_y_data(
    xname="g_param",
    yname="spin_stiffness",
    xfunc="x",
    yfunc="y*L**(0.23)",
    plot=True,
    xscale="linear",
    yscale="linear",
    xlabel="",
    ylabel="",
)

data2, _, _ = cl_runs.get_x_vs_y_data(
    xname="g_param",
    yname="binder",
    xfunc="x",
    yfunc="y",
    plot=True,
    xscale="linear",
    yscale="linear",
    xlabel="",
    ylabel="",
)
plt.show()
crossings0 = FS3.crossingPoints(data, dLfunc="2*L")
crossings2 = FS3.crossingPoints(data2, dLfunc="2*L")
# crossings2 = FS3.crossingPoints(data, dLfunc="4/3*L")

# print(crossings1,crossings2)

plt.plot(1/crossings0[0][:, 0], crossings0[1][:, 0], color="green", label="rho")
# plt.scatter(crossings1[0][:,0], crossings1[1][:,0], color="red", label="3/2 L")
plt.plot(1/crossings2[0][:,0], crossings2[1][:,0], color="blue", label="binder")
plt.show()
if True:
    fitFunc = cr.fitfunc(
        fstring="f((x/gc - 1)*L**(1/nu))*(1 + c*L**(-omega))",
        vars=["gc", "nu", "c", "omega"],
        polyOrder=1,
    )
else:
    fitFunc = cr.f1()

# fit_res = cr.data_fit(data, fitFunc, start_params=[1.342,1.0,10,2.0],fitSummary=True)
# fig = cr.fit_plot(data, fitFunc, fit_res, xlabel=r"$J^{\perp}$", ylabel=r"$U_2$")

plt.show()

# # x = np.linspace(5.6, 6.0, 1000)

# # start = time.perf_counter()
# # for n in range(100):
# #     cr.data_fit(data, fitFunc, polyOrder=2, start_params=[5.8, 0.75],fitSummary=False)

# # stop = time.perf_counter()
# # print(f"Time elapsed for 10000 evaluations: {stop - start} seconds")

# fit_res = cr.data_fit(data, fitFunc, polyOrder=2, start_params=[5.8, 0.75])
# params, dparams, redChi2, mesg, ierr = fit_res
# var = fitFunc.unpack(params)
# dvar = fitFunc.unpack(dparams)

# fig = cr.fit_plot(data, fitFunc, fit_res, xlabel=r"$J^{\perp}$", ylabel=r"$U_2$")

# plt.show()
