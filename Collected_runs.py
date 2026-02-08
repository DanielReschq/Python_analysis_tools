import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tomllib
from pathlib import Path
import sympy
import FS3
import sys
import matplotlib as mpl
import struct


def jackknife(x):
    n = len(x)
    if n != 0:
        xsum = np.sum(x)
        U0 = xsum / n
        U = (xsum - x) / (n - 1)
        Ubar = np.mean(U)
        err = np.sqrt(np.sum((U - Ubar) ** 2.0) * (n - 1) / (n))
        bias = (n - 1) * (Ubar - U0)
        mean = U0 - bias
        return mean, err, bias
    else:
        return 0, 0, 0


def get_parameters(filename):
    with open(filename, "rb") as f:
        data = tomllib.load(f)

    alpha = float(data["sigma"] + 2.0)
    N = (1 + int(data["bilayer"])) * int(data["Lx"]) * int(data["Ly"])

    data["alpha"] = alpha
    data["N"] = N

    return data


def get_data(filename, cutoff):
    data_file = pd.read_csv(filename, header=4, sep=";")

    bins = np.array(data_file["bin"], dtype=int)[cutoff:]
    n_bins = len(bins)
    acceptance_ratio = np.array(data_file["acceptance_ratio"], dtype=float)[cutoff:]
    energy = np.array(data_file["energy"], dtype=float)[cutoff:]
    mag = np.array(data_file["mag"], dtype=float)[cutoff:]
    mag_stag = np.array(data_file["mag_stag"], dtype=float)[cutoff:]
    mag2 = np.array(data_file["mag2"], dtype=float)[cutoff:]
    mag4 = np.array(data_file["mag4"], dtype=float)[cutoff:]
    spin_stiffness = np.array(data_file["spin_stiffness"], dtype=float)[cutoff:]
    susceptibility = np.array(data_file["susceptibility"], dtype=float)[cutoff:]
    susceptibility_stag = np.array(data_file["susceptibility_stag"], dtype=float)[
        cutoff:
    ]

    g_param_data = {}

    g_param_data["bins"] = bins
    g_param_data["acceptance_ratio"] = acceptance_ratio
    g_param_data["energy"] = energy
    g_param_data["mag"] = mag
    g_param_data["mag_stag"] = mag_stag
    g_param_data["mag2"] = mag2
    g_param_data["mag4"] = mag4
    g_param_data["spin_stiffness"] = spin_stiffness
    g_param_data["susceptibility"] = susceptibility
    g_param_data["susceptibility_stag"] = susceptibility_stag

    return g_param_data


def get_grid(filename):
    with open(filename, "rb") as f:
        # Read size_t (8 bytes, little endian)
        (n,) = struct.unpack("<Q", f.read(8))

        bonds = np.fromfile(f, dtype=np.int32, count=2 * n)

        # Read double array
        bond_strength = np.fromfile(f, dtype=np.float64, count=n)

        grid = {"n_bonds": n, "bonds": bonds, "bond_strength": bond_strength}

        return grid


def gauss_fit(x, mean, std_err):
    return (
        1.0
        / (np.sqrt(2 * np.pi) * std_err)
        * np.exp(-0.5 * ((x - mean) / std_err) ** 2)
    )


class single_run:
    """data of a single run. Gives mean + error"""

    def __init__(self, path_to_run: str, T0: bool, cutoff: int):
        self.path_to_run = path_to_run
        self.params = get_parameters(path_to_run + "/config.toml")
        self.data = get_data(path_to_run + "/output_files/output.csv", cutoff)
        self.time = pd.read_csv(
            path_to_run + "/output_files/time.csv", header=1, sep=";"
        )

        self.L = self.params["Lx"]
        self.temperature = self.params["temp"]
        self.j_pert = self.params["j_pert"]
        self.alpha = self.params["alpha"]

        if T0:
            self.g_param = self.j_pert
        else:
            self.g_param = self.temperature

        self.means_err = {}  # gives the means and errors and bias for the observable

        self.means_err["acceptance_ratio"] = jackknife(self.data["acceptance_ratio"])
        self.means_err["energy"] = jackknife(self.data["energy"])
        self.means_err["mag"] = jackknife(self.data["mag"])
        self.means_err["mag_stag"] = jackknife(self.data["mag_stag"])
        self.means_err["mag2"] = jackknife(self.data["mag2"])
        self.means_err["mag4"] = jackknife(self.data["mag4"])
        self.means_err["spin_stiffness"] = jackknife(self.data["spin_stiffness"])
        self.means_err["susceptibility"] = jackknife(self.data["susceptibility"])
        self.means_err["susceptibility_stag"] = jackknife(
            self.data["susceptibility_stag"]
        )

        self.data["binder"] = (
            5 / 2 * (1 - 1 / 3 * self.data["mag4"] / (self.data["mag2"] ** 2))
        )
        self.means_err["binder"] = jackknife(self.data["binder"])

    def plot_bins_vs_obs(self, obs_name):
        x = self.data["bins"]
        y = self.data[obs_name]
        mean = self.means_err[obs_name][0]
        err = self.means_err[obs_name][1]
        plt.plot(x, y)
        plt.plot(x, np.ones_like(x) * mean, color="red")
        plt.plot(x, np.ones_like(x) * (mean + err), color="red", ls="--")
        plt.plot(x, np.ones_like(x) * (mean - err), color="red", ls="--")
        plt.show()

    def get_spin_correletion(self, direction="X"):
        dict = pd.read_csv(
            self.path_to_run + f"/output_files/spin_corr{direction}.csv",
            header=None,
            sep=";",
            skiprows=5,
        )

        distance = []

        C_AA = []
        C_AA_err = []
        C_AB = []
        C_AB_err = []
        C_BA = []
        C_BA_err = []
        C_BB = []
        C_BB_err = []

        for i in range(1, len(dict.columns), 4):

            distance.append(i // 4)

            avg = np.mean(dict[i].to_numpy())
            err = np.std(dict[i].to_numpy()) / np.sqrt(len(dict[i].to_numpy()))
            C_AA.append(avg)
            C_AA_err.append(err)

            avg = np.mean(dict[i + 1].to_numpy())
            err = np.std(dict[i + 1].to_numpy()) / np.sqrt(len(dict[i + 1].to_numpy()))
            C_AB.append(avg)
            C_AB_err.append(err)

            avg = np.mean(dict[i + 2].to_numpy())
            err = np.std(dict[i + 2].to_numpy()) / np.sqrt(len(dict[i + 2].to_numpy()))
            C_BA.append(avg)
            C_BA_err.append(err)

            avg = np.mean(dict[i + 3].to_numpy())
            err = np.std(dict[i + 3].to_numpy()) / np.sqrt(len(dict[i + 3].to_numpy()))
            C_BB.append(avg)
            C_BB_err.append(err)

        spin_corr = {
            "distance": np.array(distance),
            "AA": (np.array(C_AA), np.array(C_AA_err)),
            "AB": (np.array(C_AB), np.array(C_AB_err)),
            "BA": (np.array(C_BA), np.array(C_BA_err)),
            "BB": (np.array(C_BB), np.array(C_BB_err)),
        }

        return spin_corr


class collected_runs:
    """collection of single runs across all L and g parameters"""

    def get_L_dict(self):
        """gives a dictionary sorted by L"""

        L_list = np.array(sorted(list(set(run.L for run in self.runs))))

        dic = {}

        for L in L_list:
            dic[L] = []

            for run in self.runs:
                if L == run.L:
                    dic[L].append(run)

            dic[L].sort(key=lambda x: (x.g_param))

        return L_list, dic

    def get_g_dict(self):
        """gives a dictionary sorted by temperature of j_pert"""

        g_list = np.array(sorted(list(set(run.g_param for run in self.runs))))

        dic = {}
        g_index = 0

        for g in g_list:
            dic[g_index] = []

            for run in self.runs:
                if g == run.g_param:
                    dic[g_index].append(run)
            dic[g_index].sort(key=lambda x: (x.L))
            g_index += 1
        return g_list, dic

    def __init__(self, path_to_run: str | list[str], T0: bool, cutoff: int):
        self.path_to_run = path_to_run
        self.T0 = T0
        self.runs = []
        self.obs_names = [
            "energy",
            "mag",
            "mag_stag",
            "mag2",
            "mag4",
            "susceptibility",
            "susceptibility_stag",
            "binder",
            "spin_stiffness",
        ]

        self.obs_labels = {
            "energy": r"$e$",
            "mag": r"$m$",
            "mag_stag": r"$m_s$",
            "mag2": r"$m^2$",
            "mag4": r"$m^4$",
            "susceptibility": r"$\chi$",
            "susceptibility_stag": r"$\chi_s$",
            "binder": r"$U_2$",
            "spin_stiffness": r"$\rho_s$",
        }

        if self.T0:
            self.obs_labels["g_param"] = r"$J^{\perp}$"
        else:
            self.obs_labels["g_param"] = r"$T$"

        if isinstance(path_to_run, list):
            for p in path_to_run:
                for L_folder in sorted(glob.glob(p + "/L*")):
                    for g_param_folder in sorted(glob.glob(L_folder + "/*")):
                        self.runs.append(single_run(g_param_folder, self.T0, cutoff))
        else:
            for L_folder in sorted(glob.glob(path_to_run + "/L*")):
                for g_param_folder in sorted(glob.glob(L_folder + "/*")):
                    self.runs.append(single_run(g_param_folder, self.T0, cutoff))

        self.runs.sort(key=lambda x: (x.L, x.g_param))

        self.L_list, self.L_dict = self.get_L_dict()
        self.g_list, self.g_dict = self.get_g_dict()

    def append_run(self, run: single_run):  #    add a single run to the collection
        self.runs.append(run)
        self.runs.sort(key=lambda x: (x.L, x.g_param))

        self.L_list, self.L_dict = self.get_L_dict()
        self.g_list, self.g_dict = self.get_g_dict()

    def write_to_FS3_file(self, path: str | list[str] = None):
        if path is None:
            path = self.path_to_run

        if isinstance(path, list):
            for single_path in path:
                self.write_to_FS3_file(single_path)
        else:
            filepath = path + "/FS3_format/"
            Path(filepath).mkdir(parents=True, exist_ok=True)
            for obs_name in self.obs_names:
                fname = filepath + obs_name + ".dat"
                f = open(fname, "w")

                L_index = 0
                for L in self.L_list:
                    f.write("#" + str(L_index) + "\n")
                    for run in self.L_dict[L]:
                        f.write(
                            "{:d}     {:2.15e}     {:2.15e}     {:2.15e}\n".format(
                                L,
                                run.g_param,
                                run.means_err[obs_name][0],
                                run.means_err[obs_name][1],
                            )
                        )
                    f.write("\n\n")
                    L_index += 1
                f.close()

    def get_x_vs_y_data(
        self,
        xname,
        yname,
        xfunc=None,
        yfunc=None,
        arg={},
        plot=False,
        xscale="linear",
        yscale="linear",
        xlabel="",
        ylabel="",
    ):
        """get table of any data x vs y.
        if x error exists: return tab[L] = [x,y,xerr,yerr].T
        else: return tab[L] = [x,y,yerr].T
        arg{} can be dictionary of additional parameters.
        write errors of arguments as "dName": error
        labels get appended to standard x and y label
        """

        tab = {}
        for L in self.L_list:
            xData = []
            yData = []
            xerrData = []
            yerrData = []
            for run in self.L_dict[L]:
                N = run.params["N"]

                if xname == "g_param":
                    xData.append(run.g_param)
                else:
                    xData.append(run.means_err[xname][0])
                    xerrData.append(run.means_err[xname][1])

                yData.append(run.means_err[yname][0])
                yerrData.append(run.means_err[yname][1])
            if xerrData:
                tab[L] = np.array([xData, yData, xerrData, yerrData]).T
            else:
                tab[L] = np.array([xData, yData, yerrData]).T

            if xfunc and not yfunc:
                yfunc = "y"

            if yfunc and not xfunc:
                xfunc = "x"

            if xfunc or yfunc:  # calculate correct error propagation

                dic = arg
                dic["L"] = L
                dic["Ns"] = N
                dic["x"] = "x"
                dic["y"] = "y"
                dic["dy"] = "dy"

                if len(tab[L][0]) == 4:
                    dic["dx"] = "dx"

                x_sym = sympy.sympify(xfunc)
                y_sym = sympy.sympify(yfunc)
                dx_sym = FS3.errorPropagation(x_sym)
                dy_sym = FS3.errorPropagation(y_sym)

                x_sym = x_sym.subs(dic)
                dx_sym = dx_sym.subs(dic)
                y_sym = y_sym.subs(dic)
                dy_sym = dy_sym.subs(dic)

                for name in dx_sym.free_symbols:
                    if str(name) not in dic.keys():
                        dic[name] = 0

                for name in dy_sym.free_symbols:
                    if str(name) not in dic.keys():
                        dic[name] = 0

                x_sym = x_sym.subs(dic)
                dx_sym = dx_sym.subs(dic)
                y_sym = y_sym.subs(dic)
                dy_sym = dy_sym.subs(dic)

                x_lambda = sympy.lambdify(("x", "y"), x_sym, "numpy")
                y_lambda = sympy.lambdify(("x", "y"), y_sym, "numpy")
                x = x_lambda(np.array(xData), np.array(yData))
                y = y_lambda(np.array(xData), np.array(yData))

                if len(tab[L][0]) == 4:
                    dx_lambda = sympy.lambdify(("x", "y", "dx", "dy"), dx_sym, "numpy")
                    dy_lambda = sympy.lambdify(("x", "y", "dx", "dy"), dy_sym, "numpy")
                    xerr = dx_lambda(
                        np.array(xData),
                        np.array(yData),
                        np.array(xerrData),
                        np.array(yerrData),
                    )
                    yerr = dy_lambda(
                        np.array(xData),
                        np.array(yData),
                        np.array(xerrData),
                        np.array(yerrData),
                    )

                    tab[L] = np.array([x, y, xerr, yerr]).T
                else:
                    dx_lambda = sympy.lambdify(("x", "y", "dy"), dx_sym, "numpy")
                    dy_lambda = sympy.lambdify(("x", "y", "dy"), dy_sym, "numpy")
                    xerr = dx_lambda(
                        np.array(xData), np.array(yData), np.array(yerrData)
                    )
                    yerr = dy_lambda(
                        np.array(xData), np.array(yData), np.array(yerrData)
                    )

                if np.any(xerr):
                    tab[L] = np.array([x, y, xerr, yerr]).T
                else:
                    tab[L] = np.array([x, y, yerr]).T

        FS3.setStyle(Llist=np.sort(list(tab), axis=0))
        if plot:
            xlabel = self.obs_labels[xname] + xlabel
            ylabel = self.obs_labels[yname] + ylabel

            fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=ylabel)
            FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
            plots = []
            labels = []
            for L in np.sort(list(tab), axis=0):
                if len(tab[L][0]) == 2:  # no errorbars
                    p = ax0.errorbar(
                        tab[L][:, 0],
                        tab[L][:, 1],
                        **FS3.errorbarStyle(L),
                        linestyle="",
                    )

                if len(tab[L][0]) == 3:  # only y errros
                    p = ax0.errorbar(
                        tab[L][:, 0],
                        tab[L][:, 1],
                        yerr=tab[L][:, 2],
                        **FS3.errorbarStyle(L),
                        linestyle="",
                    )

                if len(tab[L][0]) == 4:  # x and y errors
                    p = ax0.errorbar(
                        tab[L][:, 0],
                        tab[L][:, 1],
                        xerr=tab[L][:, 2],
                        yerr=tab[L][:, 3],
                        **FS3.errorbarStyle(L),
                        linestyle="",
                    )

                ax0.plot(tab[L][:, 0], tab[L][:, 1], **FS3.plotStyle(L), linestyle="-")
                plots.append(p)
                labels.append(r"$L = %s$" % L)
            FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.2, 0.2, 0.18))
            fig.suptitle(r"$\alpha = %.2f $" % (self.runs[0].alpha))
            ax0.set_xscale(xscale)
            ax0.set_yscale(yscale)

            return tab, fig, ax0

        return tab, None, None

    def select(self, LRange=[], gRange=[]):

        if len(LRange) == 0:
            LRange = np.array([-sys.maxsize - 1, sys.maxsize])
        elif len(LRange) == 1:
            LRange = np.array([LRange[0], sys.maxsize])
        else:
            LRange = np.array(LRange)

        if len(gRange) == 0:
            gRange = np.array([float("-inf"), float("inf")])
        elif len(gRange) == 1:
            gRange = np.array([gRange[0], float("inf")])
        else:
            gRange = np.array(gRange)

        self.L_list, self.L_dict = self.get_L_dict()
        self.g_list, self.g_dict = self.get_g_dict()

        self.L_list = self.L_list[
            np.where((self.L_list >= LRange[0]) & (self.L_list <= LRange[1]))
        ]
        self.g_list = self.g_list[
            np.where((self.g_list >= gRange[0]) & (self.g_list <= gRange[1]))
        ]

        self.L_dict = {}

        for L in self.L_list:
            self.L_dict[L] = []
            for run in self.runs:
                if L == run.L and run.g_param in self.g_list:
                    self.L_dict[L].append(run)
            self.L_dict[L].sort(key=lambda x: (x.g_param))

        self.g_dict = {}
        g_index = 0

        for g in self.g_list:
            self.g_dict[g_index] = []
            for run in self.runs:
                if g == run.g_param and run.L in self.L_list:
                    self.g_dict[g_index].append(run)
                self.g_dict[g_index].sort(key=lambda x: (x.L))
            g_index += 1


############### fit data #############################


def fit_data(data, fitFunction, start_params, fitSummary=False):
    params0 = start_params + np.random.randn(len(start_params)) * 0.01

    params0 = np.hstack(
        (params0, 0.01 * np.random.randn(fitFunction.nparams() - len(start_params)))
    )

    res = FS3.fit_minimize(data, fitFunction, params=params0, **{"method": "BFGS"})

    if fitSummary:
        FS3.fitSummary(fitFunction, res)

    return res


def plot_fit(
    data, fitFunction, fitResult, xlabel, ylabel, loc_legend=(0.78, 0.02, 0.18)
):
    params, dparams, redChi2, mesg, ierr = fitResult
    var = fitFunction.unpack(params)

    plots = []
    labels = []
    fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=ylabel)
    xrange, yrange = FS3.getDataRange(data, idx=0, margin=0.05), FS3.getDataRange(
        data, idx=1, margin=0.05
    )
    yrange[1] = 1.05 * yrange[1]
    ax0.set_xlim(xrange)
    ax0.set_ylim(yrange)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)

    for L in np.sort(list(data), axis=0):
        yerr = None
        xerr = None

        if len(data[L]) == 3:
            yerr = data[L][:, 2]

        if len(data[L][0]) == 4:
            xerr = data[L][:, 2]
            yerr = data[L][:, 3]

        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            xerr=xerr,
            yerr=yerr,
            **FS3.errorbarStyle(L),
            linestyle="",
        )
        plots.append(p)
        labels.append(r"$L = %s$" % L)
        plotRange = FS3.getDataRange(data[L], idx=0, margin=0.05)
        g = np.linspace(plotRange[0], plotRange[1], 400)
        ax0.plot(g, fitFunction.func(g, L, params), **FS3.plotStyle(L))
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=loc_legend)
    # ax0.axhline(var["a"][0], color="#dddddd", zorder=-1000)
    # ax0.axvline(var["gc"], color="#dddddd", zorder=-1000)

    return fig


def plot_DataCollapse(
    dataCollapse, fitFunction, fitResult, xlabel="", ylabel="", insetFigure=None, loc=3
):
    params, dparams, redChi2, mesg, ierr = fitResult
    var = fitFunction.unpack(params)

    xmin = float("inf")
    xmax = -float("inf")
    ymin = float("inf")
    ymax = -float("inf")
    for L in list(dataCollapse):
        xmin = min(np.min(dataCollapse[L][:, 0]), xmin)
        xmax = max(np.max(dataCollapse[L][:, 0]), xmax)

        ymin = min(np.min(dataCollapse[L][:, 1]), ymin)
        ymax = max(np.max(dataCollapse[L][:, 1]), ymax)

    x = np.linspace(xmin, xmax, 400)
    y = fitFunction.poly(x, params)

    if insetFigure:
        fig, ax1 = FS3.addInset(insetFigure, loc=loc, xlabel=xlabel, ylabel=ylabel)
    else:
        fig, ax1 = FS3.figure()

    ax1.plot(x, y, color="black")

    for L in np.sort(list(dataCollapse), axis=0):
        ax1.errorbar(
            dataCollapse[L][:, 0],
            dataCollapse[L][:, 1],
            yerr=dataCollapse[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle="",
        )
        # ax1.axhline(var["a"][0], color="#dddddd")
        # ax1.axvline(0, color="#dddddd")

    return fig


def resample(data):  # resample data with Gaussian noise according to the errorbars
    dataSample = {}
    for L in np.sort(list(data), axis=0):
        n = len(data[L])
        if len(data[L][0]) == 3:  # only y errors
            dataSample[L] = np.zeros((n, 3))
            dataSample[L][:, 0] = data[L][:, 0]
            dataSample[L][:, 1] = data[L][:, 1] + np.random.randn(n) * data[L][:, 2]
            dataSample[L][:, 2] = data[L][:, 2]
        elif len(data[L][0]) == 4:  # x and y errors
            dataSample[L] = np.zeros((n, 4))
            dataSample[L][:, 0] = data[L][:, 0] + np.random.randn(n) * data[L][:, 2]
            dataSample[L][:, 1] = data[L][:, 1] + np.random.randn(n) * data[L][:, 3]
            dataSample[L][:, 2] = data[L][:, 2]
            dataSample[L][:, 3] = data[L][:, 3]
    return dataSample


def scatter_histograms(xData, yData, chi2, num_bins, maxChi, xlabel=None, ylabel=None):
    fig = plt.figure()
    grid_space = mpl.gridspec.GridSpec(
        2, 2, wspace=0.1, hspace=0.1, height_ratios=[1, 1], figure=fig
    )

    ax_scatter = fig.add_subplot(grid_space[1, 0])
    ax_xhist = fig.add_subplot(grid_space[0, 0], sharex=ax_scatter)
    ax_yhist = fig.add_subplot(grid_space[1, 1], sharey=ax_scatter)
    # ax_cbar = fig.add_subplot(grid_space[2, 0])

    ax_xhist.tick_params(
        axis="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False
    )
    ax_yhist.tick_params(
        axis="both", labelbottom=True, labeltop=False, labelleft=False, labelright=True
    )

    index_small_chi = np.where(chi2 < maxChi)[0]

    ax_scatter.scatter(xData, yData, alpha=0.2, color="blue")
    xData = xData[index_small_chi]
    yData = yData[index_small_chi]

    xmin, xmax = min(xData), max(xData)
    ymin, ymax = min(yData), max(yData)

    scatter = ax_scatter.scatter(xData, yData, alpha=0.5, color="red")

    # cbar = plt.colorbar(scatter, cax=ax_cbar, orientation="horizontal")
    # cbar.set_label(r"$\chi^2$", loc="center")
    # ax_cbar.tick_params(axis="y", labelleft=True, labelright=False)
    ax_scatter.set_xlim([xmin, xmax])
    ax_scatter.set_ylim([ymin, ymax])
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    # fig.subplots_adjust(0,0,1,1,0,0)
    numx, binx, patchesx = ax_xhist.hist(xData, num_bins, [xmin, xmax], density=True)
    numy, biny, patchesy = ax_yhist.hist(
        yData, num_bins, [ymin, ymax], density=True, orientation="horizontal"
    )

    midppointsx = 0.5 * (binx[:-1] + binx[1:])
    midppointsy = 0.5 * (biny[:-1] + biny[1:])

    paramsx, pcovx = sp.optimize.curve_fit(
        gauss_fit, midppointsx, numx, p0=[np.mean([xmin, xmax]), 0.01]
    )
    x = np.linspace(xmin, xmax, 400)
    ax_xhist.plot(x, gauss_fit(x, *paramsx), color="r")

    paramsy, pcovy = sp.optimize.curve_fit(
        gauss_fit, midppointsy, numy, p0=[np.mean([ymin, ymax]), 0.01]
    )
    y = np.linspace(ymin, ymax, 400)
    ax_yhist.plot(gauss_fit(y, *paramsy), y, color="r")

    return paramsx, paramsy

############### fit function classes #################


class f0:  # Scaling function for Binder cumulant without large finite-size effects
    def __init__(self):
        self.vars = [
            "gc",
            "nu",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g / var["gc"] - 1) * L ** (1.0 / var["nu"])) ** n
        return f


class f1:  # Scaling function for Binder cumulant with large finite-size effects
    def __init__(self):
        self.vars = [
            "gc",
            "nu",
            "omega",
            "c",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g / var["gc"] - 1.0) * L ** (1.0 / var["nu"])) ** n
        return f * (1 + var["c"] * L ** (-var["omega"]))


class f_fixed_nu:  # Scaling function for Binder cumulant with fixed nu and large finite-size effects
    def __init__(self, nu):
        self.vars = [
            "gc",
            "omega",
            "c",
        ]  # define the variables used in your scaling function
        self.nu = nu
        self.polyOrder = 3  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g - var["gc"]) * L ** (1.0 / self.nu)) ** n
        return f * (1 + var["c"] * L ** (var["omega"]))


# Scaling function
class Rg0:  # Scaling function for Renormalization without large finite-size effects
    def __init__(self):
        self.vars = ["eta"]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, U, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(U)
        for n in np.arange(nmax):
            f += var["a"][n] * (U) ** n
        f = L ** (-var["eta"]) * f
        return f


class Rg1:  # Scaling function for Renormalization with large finite-size effects
    def __init__(self):
        self.vars = [
            "eta",
            "omega",
            "c",
            "omegaPrime",
            "cPrime",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, U, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(U)
        for n in np.arange(nmax):
            f += var["a"][n] * (U / (1 + var["c"] * L ** (-var["omega"]))) ** n
        f = L ** (-var["eta"]) * f
        return f * (1 + var["cPrime"] * L ** (-var["omegaPrime"]))


class f_beta_nu:
    def __init__(self):
        self.vars = [
            "beta",
            "nu",
            "gc",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g / var["gc"] - 1) * L ** (1.0 / var["nu"])) ** n
        return f * L ** (var["beta"] / var["nu"])


class f0_dyn_z:  # Scaling function for to check dynamic exponent with rho without large finite-size effects
    def __init__(self):
        self.vars = [
            "gc",
            "nu",
            "z",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g / var["gc"] - 1) * L ** (1.0 / var["nu"])) ** n
        return f * L ** (-var["z"])


class f_general:  # Scaling function with large finite-size effects and uncertain scaling with L^kappa
    def __init__(self):
        self.vars = [
            "gc",
            "1overnu",
            "kappa",
            "omega",
            "c",
        ]  # define the variables used in your scaling function
        self.polyOrder = 0  # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var["a"] = params[len(self.vars) : self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var["a"])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var["a"][n] * ((g - var["gc"]) * L ** (var["1overnu"])) ** n
        return (
            f
            * L ** (var["kappa"] * var["1overnu"])
            * (1 + var["c"] * L ** (-var["omega"]))
        )


class fitfunc:

    def f_lambda(self):
        Ls = sympy.symbols("L")
        xs = sympy.symbols("x")
        vars_symbols = sympy.symbols(self.vars)

        p = 0
        for n in range(self.polyOrder + 1):
            a_n = vars_symbols[-self.polyOrder - 1 + n]
            p += a_n * xs**n

        f = sympy.Lambda(xs, p)

        ns = {"f": f, "L": Ls, "x": xs}
        for name in self.vars:
            ns[name] = vars_symbols[self.vars.index(name)]

        f = sympy.sympify(self.fstring, locals=ns)

        return sympy.lambdify((xs, Ls, *vars_symbols), f, modules="numpy")

    def __init__(self, fstring, vars, polyOrder):
        self.vars = vars  # list of variable names in your scaling function
        self.polyOrder = (
            polyOrder  # the polyomial expansion order in your scaling function
        )
        self.fstring = fstring

        for n in range(polyOrder + 1):
            self.vars.append("a%d" % n)

        self.fitfunc_lambda = self.f_lambda()

    def nparams(self):
        return len(self.vars)

    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        return var

    def func(self, x, L, params):
        return self.fitfunc_lambda(x, L, *params)

    def poly(self, x, params):
        f = 0
        for n in range(self.polyOrder + 1):
            index = self.nparams() - (self.polyOrder + 1) + n
            f += params[index] * x**n
        return f