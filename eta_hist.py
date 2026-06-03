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


def correct_hist_df(fitFunc, hist_df):
    mask = hist_df["omega"] < 0

    hist_df_corrected = hist_df.copy()

    hist_df_corrected.loc[mask, "kappa"] = hist_df_corrected.loc[mask, "omega"] + hist_df_corrected.loc[mask, "kappa"]
    hist_df_corrected.loc[mask, "dkappa"] = np.sqrt(
        hist_df_corrected.loc[mask, "dkappa"] ** 2 + hist_df_corrected.loc[mask, "domega"] ** 2
    )

    hist_df_corrected.loc[mask, "omega"] = -hist_df_corrected.loc[mask, "omega"]

    # polynomial coefficients
    for n in range(1, fitFunc.polyOrder + 1):
        a_col = f"a{n}"
        da_col = f"da{n}"

        hist_df_corrected.loc[mask, a_col] = hist_df_corrected.loc[mask, "c"] * hist_df_corrected.loc[mask, a_col]

        hist_df_corrected.loc[mask, da_col] = np.sqrt(
            (hist_df_corrected.loc[mask, a_col] * hist_df_corrected.loc[mask, "dc"]) ** 2
            + (hist_df_corrected.loc[mask, "c"] * hist_df_corrected.loc[mask, da_col]) ** 2
        )

    hist_df_corrected.loc[mask, "c"] = 1 / hist_df_corrected.loc[mask, "c"]
    hist_df_corrected.loc[mask, "dc"] = hist_df_corrected.loc[mask, "dc"] * hist_df_corrected.loc[mask, "c"] ** 2

    print("corrected for negative omega")
    return hist_df_corrected

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
            [-1.0,1.0],
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
    "gc": np.array([9.985, 5.802, 4.33, 3.595, 3.1835, 2.942, 2.796, 2.703]),
    "nu": np.array([0.745, 0.736, 0.71, 0.71, 0.706, 0.706, 0.706, 0.706]),
    "z": np.array([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]),
    "eta": np.array([1.33, 1.30, 1.25, 1.1, 1.0, 1.0, 1.0, 1.0]),
    # "chiMax":  np.array([100000.0, 100000.0, 150000.0, 20000, 8000.0]),
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}

fitFunc = cr.fitfunc(
    # fstring="f(x)*L**(-z)",
    # fstring="f(x/(1+cU*L**(-omegaU)))*L**(-z)",
    fstring="f(x/(1+c*L**(-omega)))*L**(kappa)",
    vars=["kappa", "c", "omega"],
    # vars=["z"],
    polyOrder=3,
)

params0 = np.random.randn(fitFunc.nparams()) * 0.0001

dic = parameter_dictT

for alpha in [1.5, 2.0, 2.25, 2.5, 2.75, 3.0]:
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
            LRange = [32, 128]
            gRange = [9.93, 10.001]

            x_steps = 0.005
            y_steps = 10.0
            xRange = [0.66, 0.675]
            yRange = [0, 75.0]
            chi2max = 50.0

            params0[0] = 2.666  # 2*beta/nu-4

        case 2.0:
            LRange = [32, 128]
            gRange = [5.788, 5.8005]

            x_steps = 0.02
            y_steps = 10.0
            xRange = [0.64, 0.68]
            yRange = [0, 30.0]
            chi2max = 20.0

            params0[0] = 2.666  # 2*beta/nu-4

        case 2.5:
            LRange = [32, 128]
            gRange = [4.3225, 4.335]

            x_steps = 0.01
            y_steps = 5.0
            xRange = [0.6, 0.625]
            yRange = [0, 15.0]
            chi2max = 20.0

            params0[0] = 2.776  # 2*beta/nu-4

        case 3.0:
            LRange = [32, 128]
            gRange = [3.5885, 3.598]

            x_steps = 0.1
            y_steps = 10.0
            xRange = [0.53, 0.58]
            yRange = [0, 30.0]
            chi2max = 20.0

            params0[0] = 2.888  # 2*beta/nu-4

        case 3.5:
            LRange = [32, 128]
            gRange = [3.178, 3.1885]

            x_steps = 0.05
            y_steps = 5.0
            xRange = [0.525, 0.54]
            yRange = [0, 20.0]
            chi2max = 20.0

            params0[0] = 2.94  # 2*beta/nu-4

        case 4.0:
            LRange = [32, 128]
            gRange = [2.935, 2.948]

            x_steps = 0.01
            y_steps = 10.0
            xRange = [0.51, 0.53]
            yRange = [0, 30.0]
            chi2max = 20.0

            params0[0] = 3.0  # 2*beta/nu-4

        case 4.5:
            LRange = [32, 96]
            gRange = [2.79, 2.80]

            x_steps = 0.01
            y_steps = 5.0
            xRange = [0.495, 0.535]
            yRange = [0, 25.0]
            chi2max = 20.0

            params0[0] = 3.0  # 2*beta/nu-4

        case 5.0:
            LRange = [32, 96]
            gRange = [2.70, 2.706]

            x_steps = 0.1
            y_steps = 10.0
            xRange = [0.5, 0.56]
            yRange = [0, 30.0]
            chi2max = 20.0

            params0[0] = 3.0  # 2*beta/nu-4

    cl_runs.select(LRange=dic["LRange"][index], gRange=dic["gRange"][index])

    data, eta_fig, eta_ax = cl_runs.get_x_vs_y_data(
        xname="binder",
        yname="mag2",
        yfunc="y/L**4",
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
        xlabel=cl_runs.obs_labels["binder"],
        ylabel=cl_runs.obs_labels["mag2"],
        # loc_legend=(0.02, 0.02, 0.18),
        loc_legend=(0.25, 0.6, 0.18),
    )
    fig.suptitle(r"$\alpha = %.2f $" % (alpha))

    plt.show()

    if False: # hisogram of fit parameters
        params_list = []
        dparams_list = []
        chi_list = []

        for i in range(10000):
            if i % 1000 == 0:
                print(alpha, i)
            params0 = dic["params0"][index].copy()
            params0[0] += 0.001 * np.random.randn()  # 2*beta/nu-4
            # params0[1] += 0.001 * np.random.randn()  # c
            # params0[2] += 0.001 * np.random.randn()  # omega
            # params0[3] += 0.01 * np.random.randn()  # a0
            # params0[4] += 0.01 * np.random.randn()  # a1
            # params0[5] += 0.01 * np.random.randn()  # a2
            # params0[6] += 0.01 * np.random.randn()  # a3

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
        hist_df.to_csv("hist_beta%s_alpha%.2f_unbias.csv" % ("J", alpha))

    if False: #plotting

        # fit_res = cr.fit_data(cr.resample(data), fitFunc, start_params=params0, fitSummary=True)
        # params, dparams, redChi2, mesg, ierr = fit_res
        # var = fitFunc.unpack(params)
        # dvar = fitFunc.unpack(dparams)

        # fig, ax = cr.plot_fit(
        #     data,
        #     fitFunc,
        #     params,
        #     xlabel=cl_runs.obs_labels["binder"],
        #     ylabel=r"$M_s^2$",
        #     # loc_legend=(0.02, 0.02, 0.18),
        #     loc_legend=(0.25, 0.6, 0.18),
        # )
        # fig.suptitle(r"$\alpha = %.2f $" % (alpha))

        # plt.show()

        hist_df = pd.read_csv("hist_beta%s_alpha%.2f_unbias.csv" % ("T", alpha))
        #hist_df2 = pd.read_csv("/home/daniel/Master_thesis/DataHistJ/Old/hist_etaJ_alpha%.2f.csv" % (alpha))
        

        #hist_df_corrected = correct_hist_df(fitFunc, hist_df)
        #mask = hist_df["omega"] < 0.0

        chi_data = hist_df["chi2"]
        imin = np.argmin(hist_df["chi2"])

        keys = hist_df.keys()[2::2]
        print(keys)

        # fig, ax = plt.subplots(len(keys), 1, sharex=True, figsize=(6, 16))
        # fig.suptitle(r"$\alpha = %.2f $" % (alpha))
        # for i in range(len(keys)):
        #     ax[i].scatter(hist_df["chi2"], hist_df[keys[i]], c="blue", alpha=0.5)
        #     #ax[i].scatter(hist_df2["chi2"], -hist_df2[keys[i]], c="red", alpha=0.5)

        # fig, ax = plt.subplots(1, 1)

        # sc = ax.scatter(
        #     (-hist_df["kappa"]+4)*0.5,
        #     hist_df["chi2"],
        #     c=np.clip(hist_df["chi2"], 0, chi2max),
        #     cmap="magma",
        #     vmin=0,
        # )

        # sc = ax.scatter(
        #     (hist_df2["kappa"]+4)*0.5,
        #     hist_df2["chi2"],
        #     c=np.clip(hist_df2["chi2"], 0, chi2max),
        #     cmap="Blues",
        #     vmin=0,
        # )

        # cbar = fig.colorbar(sc)
        # cbar.set_label(r"$\chi^2$")
        # ticks = np.linspace(0, chi2max, int(chi2max) + 1)
        # cbar.set_ticks([0, np.floor(chi2max / 2), chi2max])
        # cbar.set_ticklabels(["0", str(np.floor(chi2max / 2)), ">" + str(chi2max)])

        x_fit, y_fit, x_min, y_min = cr.scatter_histograms(
            xData=(-hist_df["kappa"]+4)*0.5,
            yData=hist_df["chi2"],
            x_steps=x_steps,
            y_steps=y_steps,
            x_range=xRange,
            y_range=yRange,
            chi2=hist_df["chi2"],
            num_bins=50,
            xlabel=r"$\beta/\nu$",
            ylabel=r"$\chi^2$",
        )

        plt.savefig("beta%s_alpha%.2f_unbias.pdf" % ("J", alpha))

        #plt.show()
