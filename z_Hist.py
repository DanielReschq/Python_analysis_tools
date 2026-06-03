import numpy as np
import matplotlib.pyplot as plt
import Collected_runs_copy as cr
import pandas as pd
import FS3


def correct_hist_df(fitFunc, hist_df):
    mask = hist_df["omega"] < 0

    hist_df_corrected = hist_df.copy()

    hist_df_corrected.loc[mask, "z"] = hist_df_corrected.loc[mask, "omega"] + hist_df_corrected.loc[mask, "z"]
    hist_df_corrected.loc[mask, "dz"] = np.sqrt(
        hist_df_corrected.loc[mask, "dz"] ** 2 + hist_df_corrected.loc[mask, "domega"] ** 2
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
    #fstring="f(x/(1+c*L**(-omega)))*L**(-z)",
    fstring="f(x)*L**(-z)*(1+c*L**(-omega))",
    vars=["z", "c", "omega"],
    # vars=["z"],
    polyOrder=3,
)

params0 = np.random.randn(fitFunc.nparams()) * 0.0001

dic = parameter_dictJ

for alpha in [3.0, 3.5, 4.0, 4.5, 5.0]:
    index = list(dic["alpha"]).index(alpha)
    path = [
        dic["path_to_data"] + "runs" + dic["sigma"][index],
        # "/home/daniel/Master_thesis/Data/runs_Js250",
        # "/home/daniel/Old_master_thesis_folders/master_thesis/runs_LEO4_new/runsJ_s00"
    ]

    path = [dic["path_to_data"] + "runs" + dic["sigma"][index]]
    cl_runs = cr.collected_runs(path, True, cutoff=100)

    match alpha:
        case 1.5:
            LRange = [32, 128]
            gRange = [9.93, 10.001]

            x_steps = 0.02
            y_steps = 0.1
            xRange = [-0.05, 0.05]
            yRange = [0, 0.3]
            chi2max = 50.0

            params0[0] = -0.0  # z
            params0[1] = -1.25  # c
            params0[2] = 0.18  # omega
            params0[3] = 0.8  # a0
            params0[4] = 8.0  # a1
            params0[5] = -12.0  # a2
            params0[6] = 15.0  # a3

        case 2.0:
            LRange = [32, 128]
            gRange = [5.788, 5.8005]

            x_steps = 0.1
            y_steps = 0.2
            xRange = [0.2, 0.5]
            yRange = [0, 1.0]
            chi2max = 20.0

            params0[0] = 0.5  # z
            params0[1] = -1.1  # c
            params0[2] = 0.15  # omega
            params0[3] = 1.0  # a0
            params0[4] = -1.0  # a1
            params0[5] = 30.0  # a2
            params0[6] = -50.0  # a3

        case 2.5:
            LRange = [32, 128]
            gRange = [4.3225, 4.335]

            x_steps = 0.1
            y_steps = 0.5
            xRange = [0.7, 0.85]
            yRange = [0, 1.0]
            chi2max = 20.0

            params0[0] = 0.75  # z
            params0[1] = -1.5  # c
            params0[2] = 0.3  # omega
            params0[3] = 0.25  # a0
            params0[4] = 5.0  # a1
            params0[5] = 2.5  # a2
            params0[6] = 2.5  # a3

        case 3.0:
            LRange = [32, 128]
            gRange = [3.5885, 3.598]

            x_steps = 0.1
            y_steps = 0.5
            xRange = [0.9, 1.2]
            yRange = [0, 1.0]
            chi2max = 20.0

            params0[0] = 1.0  # z
            params0[1] = -1.1  # c
            params0[2] = 0.2  # omega
            params0[3] = -5  # a0
            params0[4] = 40.0  # a1
            params0[5] = -60.0  # a2
            params0[6] = 65.0  # a3

        case 3.5:
            LRange = [32, 128]
            gRange = [3.178, 3.1885]

            x_steps = 0.1
            y_steps = 0.5
            xRange = [0.9, 1.0]
            yRange = [0, 1.0]
            chi2max = 20.0

            params0[0] = 1.0  # z
            params0[1] = 0.75  # c
            params0[2] = 3.0  # omega
            params0[3] = -0.05  # a0
            params0[4] = 4.0  # a1
            params0[5] = -5.0  # a2
            params0[6] = 5.0  # a3

        case 4.0:
            LRange = [32, 128]
            gRange = [2.935, 2.948]

            x_steps = 0.1
            y_steps = 0.5
            xRange = [0.9, 1.1]
            yRange = [0, 1.0]
            chi2max = 20.0

            params0[0] = 1.0  # z
            params0[1] = 0.5  # c
            params0[2] = 3.0  # omega
            params0[3] = -0.05  # a0
            params0[4] = 4.0  # a1
            params0[5] = -5.0  # a2
            params0[6] = 5.0  # a3

        case 4.5:
            LRange = [32, 96]
            gRange = [2.79, 2.80]

            x_steps = 0.1
            y_steps = 1.0
            xRange = [0.9, 1.1]
            yRange = [-5, 5]
            chi2max = 20.0

            params0[0] = 1.0  # z

        case 5.0:
            LRange = [32, 96]
            gRange = [2.70, 2.706]

            x_steps = 0.1
            y_steps = 1.0
            xRange = [0.9, 1.1]
            yRange = [-5, 5]
            chi2max = 20.0

            params0[0] = 1.0  # z

    cl_runs.select(LRange=LRange, gRange=gRange)

    data, z_fig, z_ax = cl_runs.get_x_vs_y_data(
        xname="binder",
        yname="spin_stiffness",
        plot=False,
    )

    if False:
        params_list = []
        dparams_list = []
        chi_list = []

        for i in range(100):
            if i % 100 == 0:
                print(alpha, i)
            params0[0] += 0.01 * np.random.randn()  # z
            params0[1] += 0.001 * np.random.randn()  # c
            params0[2] += 0.001 * np.random.randn()  # omega
            params0[3] += 0.01 * np.random.randn()  # a0
            params0[4] += 0.01 * np.random.randn()  # a1
            params0[5] += 0.01 * np.random.randn()  # a2
            params0[6] += 0.01 * np.random.randn()  # a3

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
        hist_df.to_csv("hist_z%s_alpha%.2f_test.csv" % ("J", alpha))

    else:

        # fit_res = cr.fit_data(cr.resample(data), fitFunc, start_params=params0, fitSummary=True)
        # params, dparams, redChi2, mesg, ierr = fit_res
        # var = fitFunc.unpack(params)
        # dvar = fitFunc.unpack(dparams)

        # fig, ax = cr.plot_fit(
        #     data,
        #     fitFunc,
        #     params,
        #     xlabel=cl_runs.obs_labels["binder"],
        #     ylabel=cl_runs.obs_labels["spin_stiffness"],
        #     # loc_legend=(0.02, 0.02, 0.18),
        #     loc_legend=(0.25, 0.6, 0.18),
        # )
        # fig.suptitle(r"$\alpha = %.2f $" % (alpha))

        # plt.show()

        hist_df = pd.read_csv("hist_z%s_alpha%.2f_unbias.csv" % ("J", alpha))

        hist_df = correct_hist_df(fitFunc, hist_df)
        mask = hist_df["omega"] < 0.0

        chi_data = hist_df["chi2"]
        imin = np.argmin(hist_df["chi2"])

        keys = hist_df.keys()[2::2]
        print(keys)

        # fig, ax = plt.subplots(len(keys), 1, sharex=True, figsize=(6, 16))
        # fig.suptitle(r"$\alpha = %.2f $" % (alpha))
        # for i in range(len(keys)):
        #     ax[i].scatter(hist_df["z"], hist_df[keys[i]], c="blue", alpha=0.5)
        #     ax[i].scatter(
        #         hist_df_corrected["z"][mask],
        #         hist_df_corrected[keys[i]][mask],
        #         c="red",
        #         alpha=0.5,
        #     )

        # fig, ax = plt.subplots(1, 1)

        # sc = ax.scatter(
        #     hist_df["z"],
        #     hist_df["omega"],
        #     c=np.clip(hist_df["chi2"], 0, chi2max),
        #     cmap="magma",
        #     vmin=0,
        # )

        # cbar = fig.colorbar(sc)
        # cbar.set_label(r"$\chi^2$")
        # ticks = np.linspace(0, chi2max, int(chi2max) + 1)
        # cbar.set_ticks([0, np.floor(chi2max / 2), chi2max])
        # cbar.set_ticklabels(["0", str(np.floor(chi2max / 2)), ">" + str(chi2max)])

        x_fit, y_fit, x_min, y_min = cr.scatter_histograms(
            xData=hist_df["z"],
            yData=hist_df["omega"],
            x_steps=x_steps,
            y_steps=y_steps,
            x_range=xRange,
            y_range=yRange,
            chi2=hist_df["chi2"],
            num_bins=50,
            xlabel=r"$z$",
            ylabel=r"$\omega$",
        )

        #plt.savefig("z_%s_alpha%.2f_unbias.pdf" % ("J", alpha))
        plt.show()
