import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tomllib

# from matplotlib.lines import Line2D
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


class single_run:
    """data of a single run. Gives mean + error"""

    def __init__(self, path_to_run: str, T0: bool, cutoff: int):
        self.path_to_run = path_to_run
        self.params = get_parameters(path_to_run + "/config.toml")
        self.data = get_data(path_to_run + "/output_files/output.csv", cutoff)

        # self.data["energy"] = self.data["energy"] / self.params["N"]
        # self.data["mag"] = self.data["mag"] / self.params["N"]
        # self.data["mag_stag"] = self.data["mag_stag"] / self.params["N"]
        # self.data["mag2"] = self.data["mag2"] / (self.params["N"] ** 2)
        # self.data["mag4"] = self.data["mag4"] / (self.params["N"] ** 4)
        # self.data["spin_stiffness"] = self.data["spin_stiffness"] / 2.0
        # self.data["susceptibility"] = self.data["susceptibility"] / self.params["N"]
        # self.data["susceptibility_stag"] = (
        #     self.data["susceptibility_stag"] / self.params["N"]
        # )

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

    def get_Spin_correletion(self):
        dict = pd.read_csv(
            self.path_to_run + "/output_files/spin_corrX.csv",
            header=None,
            sep=";",
            skiprows=5,
        )

        C_AA = []
        C_AA_err = []
        C_AB = []
        C_AB_err = []
        C_BA = []
        C_BA_err = []
        C_BB = []
        C_BB_err = []

        for i in range(1, len(dict.columns), 4):

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

        L_list = sorted(list(set(run.L for run in self.runs)))

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

        g_list = sorted(list(set(run.g_param for run in self.runs)))

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

    def __init__(self, path_to_run: str, T0: bool, cutoff: int):
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

    def write_to_FS3_file(self, suffix=""):
        fname = ""
        for obs_name in self.obs_names:
            if self.T0:
                fname = "../dataJ/" + obs_name + suffix + ".dat"
            else:
                fname = "../dataT/" + obs_name + suffix + ".dat"
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
            f.close()

    def get_param_vs_obs(self, obs_name):
        tab = {}
        for L in self.L_list:
            x = []
            y = []
            yerr = []
            for run in self.L_dict[L]:
                x.append(run.g_param)
                y.append(run.means_err[obs_name][0])
                yerr.append(run.means_err[obs_name][1])

            tab[L] = np.array(x), np.array(y), np.array(yerr)

        return tab

    def plot_obs(
        self,
        obs_name,
        x_factor=lambda L, N: 1.0,
        y_factor=lambda L, N: 1.0,
        xscale="linear",
        yscale="linear",
    ):
        for L in self.L_list:
            x = []
            y = []
            yerr = []
            for run in self.L_dict[L]:
                N = run.params["N"]
                x.append(run.g_param * x_factor(L, N))
                y.append(run.means_err[obs_name][0] * y_factor(L, N))
                yerr.append(run.means_err[obs_name][1] * y_factor(L, N))

            plt.errorbar(x, y, yerr, marker="o", linestyle="-", label=f"L={L}")
        plt.legend()
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.title(r"$\alpha = %.2f $" % (self.runs[0].alpha))
        plt.show()


############### fit function classes #################


class f0:  # Scaling function for Binder cumulant without large finite-size effects
    def __init__(self):
        self.vars = ["gc", "nu"]  # define the variables used in your scaling function
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
            f += var["a"][n] * ((g - var["gc"]) * L ** (1.0 / var["nu"])) ** n
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
            f += var["a"][n] * ((g - var["gc"]) * L ** (1.0 / var["nu"])) ** n
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
