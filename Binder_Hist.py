import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import Collected_runs_copy as cr
import pandas as pd
import FS3

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
        "figure.autolayout": True,
    }
)


def U2(mag2, mag4):
    return float(5 / 2 * (1 - 1 / 3 * mag4 / (mag2**2)))


def bootstrap_data(runs):
    tab = {}
    for L in runs.L_list:
        xData = []
        yData = []
        for run in cl_runs.L_dict[L]:
            xData.append(run.params["j_pert"])
            n = len(run.data["bins"])
            idx = np.random.default_rng().integers(0, n, n)
            rho = run.data["spin_stiffness"][idx]
            # mag2 = run.data["mag2"][idx]
            # mag4 = run.data["mag4"][idx]
            # yData.append(U2(np.mean(mag2),np.mean(mag4)))
            yData.append(np.mean(rho))
        tab[L] = np.array([xData, yData]).T
    return tab


parameter_dictT = {
    "path_to_data": "/home/daniel/Master_thesis/Data/Leo4Data/",
    "alpha": np.array([1.5, 2.0, 2.25, 2.5, 2.75, 3.0]),
    "sigma": ["_Tsm050", "_Ts000", "_Ts025", "_Ts050", "_Ts075", "_Ts100"],
    "gRange": [
        np.array([2.425, 2.455]),
        np.array([1.3, 1.348]),
        np.array([1.094, 1.1]),
        np.array([0.914, 0.924]),
        np.array([0.774, 0.786]),
        np.array([0.6, 0.67]),
    ],
    "LRange": np.array(
        [[64, 256], [64, 256], [64, 256], [64, 256], [64, 256], [64, 256]]
    ),
    "params0": np.array(
        [
            [2.436, 1.0],
            [1.34227, 1.01],
            [1.097, 1.05],
            [0.917, 1.41],
            [0.776, 2.0],
            [0.66, 1.8],
        ]
    ),
    "xsteps": np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "ysteps": np.array([0.0005, 0.001, 0.001, 0.0005, 0.0005, 0.0005]),
    "xRange": np.array(
        [
            [0.96, 1.02],
            [0.98, 1.01],
            [0.9, 1.1],
            [0.9, 1.6],
            [0.9, 2.1],
            [0.9, 3.0],
        ]
    ),
    "yRange": np.array(
        [
            [2.435, 2.437],
            [1.342, 1.348],
            [1.094, 1.1],
            [0.914, 0.924],
            [0.774, 0.786],
            [0.6, 0.67],
        ]
    ),
    "chi2max": np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
    # "gc": np.array([2.436, 1.34227, 1.097, 0.917, 0.776, 0.66]),
    # "nu": np.array([0.976, 1.01, 1.05, 1.41, 2.0, 1.8]),
    # "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # "eta": np.array([0.997, 0.935, 0.8, 0.66, 0.5, 0.4]),
    # "chiMax": [9.0, 100.9, 800.4, 100.4, 30000, 50000],
    # "rho_L_power": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # "JRange": [np.array([4.9,5.2]),np.array([2.89,2.90]),np.array([2.15,2.17]),np.array([0,2.18]),np.array([0,2.18])],
    # "J0": [5.1,2.895,2.165,1.89,2.17],
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}

parameter_dictJ = {
    "path_to_data": "/home/daniel/Master_thesis/Data/Leo4Data/",
    "alpha": np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
    "sigma": [
        "_Jsm050",
        "_Js000",
        "_Js050",
        "_Js100",
        "_Js150",
        "_Js200",
        "_Js250",
        "_Js300",
    ],
    "gRange": [
        np.array([9.955, 10.001]),
        np.array([5.788, 5.8005]),
        np.array([4.3225, 4.335]),
        np.array([3.5885, 3.6015]),
        np.array([3.1775, 3.1885]),
        np.array([2.935, 2.948]),
        np.array([2.789, 2.805]),
        np.array([2.698, 2.707]),
    ],
    "LRange": np.array(
        [
            [32, 128],
            [32, 128],
            [32, 128],
            [32, 128],
            [32, 128],
            [32, 128],
            [32, 96],
            [32, 96],
        ]
    ),
    "params0": np.array(
        [
            [9.987, 0.75],
            [5.797, 0.73],
            [4.33, 0.72],
            [3.595, 0.71],
            [3.183, 0.71],
            [2.942, 0.71],
            [2.796, 0.71],
            [2.704, 0.7],
        ]
    ),
    "xsteps": np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "ysteps": np.array([0.0005, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]),
    "xRange": np.array(
        [
            [0.96, 1.02],
            [0.71, 0.745],
            [0.705, 0.735],
            [0.699, 0.721],
            [0.699, 0.731],
            [0.689, 0.721],
            [0.689, 0.731],
            [0.689, 0.731],
        ]
    ),
    "yRange": np.array(
        [
            [2.435, 2.437],
            [5.795, 5.798],
            [4.33, 4.331],
            [3.59449, 3.59551],
            [3.183, 3.184],
            [2.941, 2.943],
            [2.7955, 2.7965],
            [2.7035, 2.7045],
        ]
    ),
    "chi2max": np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
    # "gc": np.array([9.985, 5.802, 4.33, 3.595, 3.1835, 2.942, 2.796, 2.703]),
    # "nu": np.array([0.745, 0.736, 0.71, 0.71, 0.706, 0.706, 0.706, 0.706]),
    # "z": np.array([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]),
    # "eta": np.array([1.33, 1.30, 1.25, 1.1, 1.0, 1.0, 1.0, 1.0]),
    # "chiMax":  np.array([100000.0, 100000.0, 150000.0, 20000, 8000.0]),
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}

fitFunc = cr.fitfunc(
    fstring="f((x/gc-1)*L**(1/nu))*(1 + c*L**(-omega))",
    # fstring="f((x/gc - 1)*L**(1/nu))",
    vars=["gc", "nu", "c", "omega"],
    # vars=["gc", "nu"],
    polyOrder=3,
)

params0 = np.random.randn(fitFunc.nparams()) * 0.0001

dic = parameter_dictT

for alpha in [1.5]:  # dic["alpha" ]:
    index = list(dic["alpha"]).index(alpha)
    path = [
        dic["path_to_data"] + "runs" + dic["sigma"][index],
        # "/home/daniel/Master_thesis/Data/runs_Js250",
        # "/home/daniel/Old_master_thesis_folders/master_thesis/runs_LEO4_new/runsJ_s00"
    ]

    path = [dic["path_to_data"] + "runs" + dic["sigma"][index]]
    cl_runs = cr.collected_runs(path, False, cutoff=100)

    match alpha:
        case 1.5:
            # LRange = [32, 128]
            LRange = dic["LRange"][index]
            gRange = [9.93, 10.001]

            x_steps = 0.02
            y_steps = 0.0005
            xRange = [0.96, 1.02]  # T
            yRange = [2.435, 2.437]  # T
            # xRange = [0.739, 0.771] #J
            # yRange = [9.985, 9.989]#J
            chi2max = 6.0

            params0[0] = 9.987  # gc
            params0[1] = 0.75  # nu
            # params0[2] = 7.5  # c
            # params0[3] = 1.0  # omega
            # params0[4] = 0.09  # a0
            # params0[5] = -0.035  # a1
            # params0[6] = 0.008  # a2
            # params0[7] = -0.0025  # a3
        case 2.0:
            LRange = [64, 128]
            gRange = [5.788, 5.8005]

            x_steps = 0.01
            y_steps = 0.001
            xRange = [0.71, 0.745]
            yRange = [5.795, 5.798]
            chi2max = 6.0

            params0[0] = 5.797  # gc
            params0[1] = 0.73  # nu
            # params0[2] = 2.5  # c
            # params0[3] = 0.2  # omega
            # params0[4] = 0.15  # a0
            # params0[5] = -0.05  # a1
            # params0[6] = 0.015  # a2
            # params0[7] = -0.001  # a3
        case 2.5:
            LRange = [32, 128]
            gRange = [4.3225, 4.335]

            x_steps = 0.01
            y_steps = 0.001
            xRange = [0.705, 0.735]
            yRange = [4.33, 4.331]
            chi2max = 6.0

            params0[0] = 4.33  # gc
            params0[1] = 0.72  # nu
            # params0[2] = 1.0  # c
            # params0[3] = 0.8  # omega
            # params0[4] = 0.32  # a0
            # params0[5] = -0.1  # a1
            # params0[6] = 0.015  # a2
            # params0[7] = 0.0025  # a3

        case 3.0:
            LRange = [32, 128]
            gRange = [3.5885, 3.598]

            x_steps = 0.01
            y_steps = 0.0005
            xRange = [0.699, 0.721]
            yRange = [3.59449, 3.59551]
            chi2max = 6.0

            params0[0] = 3.595  # gc
            params0[1] = 0.7  # nu
            # params0[2] = 0.5  # c
            # params0[3] = 0.1  # omega
            # params0[4] = 0.5  # a0
            # params0[5] = -0.1  # a1
            # params0[6] = 0.004  # a2
            # params0[7] = 0.005  # a3

        case 3.5:
            LRange = [32, 128]
            gRange = [3.178, 3.1885]

            x_steps = 0.01
            y_steps = 0.0005
            xRange = [0.699, 0.731]
            yRange = [3.183, 3.184]
            chi2max = 6.0

            params0[0] = 3.183  # gc
            params0[1] = 0.71  # nu
            # params0[2] = 0.5  # c
            # params0[3] = 0.02  # omega
            # params0[4] = 0.33  # a0
            # params0[5] = -0.08  # a1
            # params0[6] = 0.0025  # a2
            # params0[7] = 0.003  # a3

        case 4.0:
            LRange = [32, 128]
            gRange = [2.935, 2.948]

            x_steps = 0.01
            y_steps = 0.0005
            xRange = [0.689, 0.721]
            yRange = [2.941, 2.943]
            chi2max = 6.0

            params0[0] = 2.942  # gc
            params0[1] = 0.71  # nu
            # params0[2] = 0.5  # c
            # params0[3] = 0.8  # omega
            # params0[4] = 0.56  # a0
            # params0[5] = -0.1  # a1
            # params0[6] = -0.005  # a2
            # params0[7] = 0.003  # a3

        case 4.5:
            LRange = [32, 96]
            gRange = [2.79, 2.80]

            x_steps = 0.01
            y_steps = 0.0005
            xRange = [0.689, 0.731]
            yRange = [2.7955, 2.7965]
            chi2max = 6.0

            params0[0] = 2.796  # gc
            params0[1] = 0.71  # nu
            # params0[2] = 10.  # c
            # params0[3] = 3.0  # omega
            # params0[4] = 0.585  # a0
            # params0[5] = -0.12  # a1
            # params0[6] = -0.008  # a2
            # params0[7] = 0.003  # a3

        case 5.0:
            LRange = [32, 96]
            gRange = [2.70, 2.706]

            x_steps = 0.01
            y_steps = 0.0005
            xRange = [0.689, 0.731]
            yRange = [2.7035, 2.7045]
            chi2max = 6.0

            params0[0] = 2.704  # gc
            params0[1] = 0.7  # nu
            # params0[2] = 2.0  # c
            # params0[3] = 1.5  # omega
            # params0[4] = 0.58  # a0
            # params0[5] = -0.12  # a1
            # params0[6] = -0.01  # a2
            # params0[7] = 0.00  # a3

    cl_runs.select(LRange=dic["LRange"][index], gRange=dic["gRange"][index])

    data, binder_fig, binder_ax = cl_runs.get_x_vs_y_data(
        xname="g_param",
        yname="binder",
        plot=False,
    )

    fit_res = cr.fit_data(cr.resample(data), fitFunc, start_params=dic["params0"][index], fitSummary=True)
    params, dparams, redChi2, mesg, ierr = fit_res
    var = fitFunc.unpack(params)
    dvar = fitFunc.unpack(dparams)

    fig, ax = cr.plot_fit(
        data,
        fitFunc,
        params,
        xlabel=cl_runs.obs_labels["g_param"],
        ylabel=r"$U_2$",
        # loc_legend=(0.02, 0.02, 0.18),
        loc_legend=(0.25, 0.6, 0.18),
    )
    fig.suptitle(r"$\alpha = %.2f $" % (alpha))

    plt.show()

    params_list = []
    dparams_list = []
    chi_list = []

    if False: # histogram

        for i in range(1):
            if i % 1000 == 0:
                print(alpha, i)

            params0 = dic["params0"][index].copy()
            params0[0] += 0.0001 * np.random.randn()  # gc
            params0[1] += 0.001 * np.random.randn()  # nu
            # params0[2] += 0.001 * np.random.randn()  # c
            # params0[3] += 0.001 * np.random.randn()  # omega
            # params0[4] += 0.0001 * np.random.randn()  # a0
            # params0[5] += 0.0001 * np.random.randn()  # a1
            # params0[6] += 0.0001 * np.random.randn()  # a2
            # params0[7] += 0.0001 * np.random.randn()  # a3

            fit_res = cr.fit_data(
                cr.resample(data), fitFunc, start_params=params0, fitSummary=False
            )
            params, dparams, redChi2, mesg, ierr = fit_res
            params_list.append(params)
            dparams_list.append(dparams)
            chi_list.append(redChi2)

        hist_data = {}
        hist_data["chi2"] = chi_list

        for i in range(len(fitFunc.vars)):
            var_name = fitFunc.vars[i]
            err_name = "d" + fitFunc.vars[i]
            hist_data[var_name] = np.array(params_list)[:, i]
            hist_data[err_name] = np.array(dparams_list)[:, i]

        hist_df = pd.DataFrame.from_dict(hist_data)
        # hist_df.to_csv("hist_binder%s_alpha%.2f_unbias.csv" % ("T", alpha))

    hist_df = pd.read_csv("hist_binder%s_alpha%.2f_unbias.csv" % ("T", alpha))
    # hist_df = pd.read_csv("/home/daniel/Master_thesis/DataHistT/N10000/hist_binder%s_alpha%.2f_bias.csv" % ("T", alpha))

    if False:  # plot
        chi_data = hist_df["chi2"]
        imin = np.argmin(hist_df["chi2"])

        keys = hist_df.keys()[2::2]
        print(keys)

        fig, ax = plt.subplots(len(keys), 1, sharex=True, figsize=(6, 16))
        for i in range(len(keys)):
            ax[i].scatter(hist_df["chi2"], hist_df[keys[i]], c="blue", alpha=0.5)

        fig, ax = plt.subplots(1, 1)

        sc = ax.scatter(
            hist_df["gc"],
            hist_df["nu"],
            c=np.clip(hist_df["chi2"], 0, chi2max),
            cmap="magma",
            vmin=0,
        )

        cbar = fig.colorbar(sc)
        cbar.set_label(r"$\chi^2$")
        ticks = np.linspace(0, chi2max, int(chi2max) + 1)
        cbar.set_ticks([0, np.floor(chi2max / 2), chi2max])
        cbar.set_ticklabels(["0", str(np.floor(chi2max / 2)), ">" + str(chi2max)])

        x_fit, y_fit, x_min, y_min = cr.scatter_histograms(
            xData=hist_df["nu"],
            yData=hist_df["gc"],
            x_steps=dic["xsteps"][index],
            y_steps=dic["ysteps"][index],
            x_range=dic["xRange"][index],
            y_range=dic["yRange"][index],
            chi2=hist_df["chi2"],
            num_bins=50,
            xlabel=r"$\nu$",
            ylabel=r"$g_c$",
        )

        plt.show()
        # plt.savefig("nu_%s_alpha%.2f_unbias.pdf" % ("J", alpha))

# betaNuInv = [0.662, 0.654, 0.612, 0.556, 0.529, 0.52, 0.516, 0.520]
# betaNuInv_min = [0.663, 0.652, 0.608, 0.557, 0.528, 0.522, 0.512, 0.519]
# betaNuInv_err = [0.003, 0.002, 0.003, 0.005, 0.004, 0.004, 0.006, 0.004]

# z = [-0.013, 0.397, 0.77, 0.953, 1.02, 1.05, 1.06, 0.939]
# z_min = [-0.015, 0.369, 0.755, 0.975, 1.05, 1.04, 0.972, 0.989]
# z_err = [0.006, 0.005, 0.01, 0.008, 0.02, 0.02, 0.06, 0.005]

# z_NoFS = [-0.296, 0.1797, 0.587, 0.848, 0.949, 0.962, 0.949, 0.942]
# z_min_NoFS = [-0.295, 0.181, 0.588, 0.849, 0.951, 0.964, 0.951, 0.942]
# z_err_NoFS = [0.0005, 0.0006, 0.0009, 0.001, 0.001, 0.001, 0.002, 0.001]
