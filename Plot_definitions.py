import FS3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
import scipy.optimize

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


def resample(data):  # resample data with Gaussian noise according to the errorbars
    dataSample = {}
    for L in np.sort(list(data), axis=0):
        n = len(data[L])
        if len(data[L][0]) == 3:  # only y errors
            dataSample[L] = np.zeros((n, 2))
            dataSample[L][:, 0] = data[L][:, 0]
            dataSample[L][:, 1] = data[L][:, 1] + np.random.randn(n) * data[L][:, 2]
        elif len(data[L][0]) == 4:  # x and y errors
            dataSample[L] = np.zeros((n, 2))
            dataSample[L][:, 0] = data[L][:, 0] + np.random.randn(n) * data[L][:, 1]
            dataSample[L][:, 1] = data[L][:, 2] + np.random.randn(n) * data[L][:, 3]
    return dataSample


def powerlaw(p, x):  # power law function p[0] * x^(-p[1])
    return p[0] * x ** (-p[1])


def fit(x, y, xerr=0, yerr=0, p0=None):  # fit to power law with errors in x and y
    if type(xerr) == int:
        xerr = np.zeros_like(x)
    if type(yerr) == int:
        yerr = np.zeros_like(y)
    err_func = lambda p, x, y, xerr, yerr: (y - powerlaw(p, x)) / np.sqrt(
        xerr**2 + yerr**2
    )

    if p0 == None:
        p0 = np.random.random(2)
    mean, cov, info, mesg, ierr = sp.optimize.leastsq(
        err_func, p0, args=(x, y, xerr, yerr), full_output=1
    )
    rchi2 = np.sum(err_func(mean, x, y, xerr, yerr) ** 2) / (len(x) - len(mean))

    return mean, np.sqrt(rchi2 * np.diag(cov)), rchi2


def plot_fit_obs(obs_name):
    data = FS3.load(
        "/home/daniel/Master_thesis_new/dataT/binder_Ts00.dat",
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )

    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    fig, ax0 = FS3.figure(xlabel=r"$T$", ylabel=r"$U_2$")
    xrange = FS3.getDataRange(data, idx=0, margin=0.05)
    # ax0.set_xlim(1.2,1.4)
    ax0.set_ylim(0, 1)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            yerr=data[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        ax0.plot(data[L][:, 0], data[L][:, 1], **FS3.plotStyle(L), linestyle="-")
        plots.append(p)
        labels.append(r"$L = %s$" % L)
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.74, 0.72, 0.18))

    data = FS3.select(data, gRange=[1.2, 1.6])
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

"""
def plot_Binder(
    parameter_dict,
    dict_index,
    polyOrder=2,
    high_FS_effect=True,
    seed0=np.random.get_state(),
    save=False,
    finite_T=True,
):
    # raw plot
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    if finite_T:
        xlabel = r"$T$"
        xlabel_red = r"$(T-T_c)\, L^{1/\nu}$"
        ylabel_red = r"$U_2/(1+cL^{-\omega})$"
        dir = "plotsT/"
    else:
        xlabel = r"$J^{\perp}$"
        xlabel_red = r"$(J^{\perp}-J^{\perp}_c)\, L^{1/\nu}$"
        ylabel_red = r"$U_2/(1+cL^{-\omega})$"
        dir = "plotsJ/"

    fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=r"$U_2$")
    xrange = FS3.getDataRange(data, idx=0, margin=0.05)
    ax0.set_xlim(data[128][0, 0], data[128][-1, 0])
    ax0.set_ylim(0, 1)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            yerr=data[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        ax0.plot(data[L][:, 0], data[L][:, 1], **FS3.plotStyle(L), linestyle="-")
        plots.append(p)
        labels.append(r"$L = %s$" % L)
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.74, 0.72, 0.18))
    ax0.set_title(r"$\alpha = %.1f$" % parameter_dict["alpha"][dict_index])
    if save:
        fig.savefig(dir + "Binder_alpha%.1f.pdf" % parameter_dict["alpha"][dict_index])

    # fitted plot
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    if high_FS_effect:
        fitFunction = f1()
    else:
        fitFunction = f0()
    fitFunction.polyOrder = polyOrder

    g0 = parameter_dict["gc"][dict_index]
    nu0 = parameter_dict["nu"][dict_index]

    np.random.set_state(seed0)
    while True:

        seed = np.random.get_state()
        params0 = np.hstack(
            (
                [g0 + np.random.randn() * 0.01, nu0 + np.random.randn() * 0.01],
                0.01 * np.random.randn(fitFunction.nparams() - 2),
            )
        )
        res = FS3.fit_minimize(data, fitFunction, params=params0, **{"method": "BFGS"})
        params, dparams, redChi2, mesg, ierr = res

        if redChi2 < parameter_dict["chiMax"][dict_index]:
            break

    output = FS3.fitSummary(fitFunction, res)

    # main panel, crossings
    var = fitFunction.unpack(params)
    fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=r"$U_2$")
    xrange, yrange = FS3.getDataRange(data, idx=0, margin=0.05), FS3.getDataRange(
        data, idx=1, margin=0.05
    )
    yrange[1] = 1.05 * yrange[1]
    ax0.set_xlim(xrange)
    ax0.set_ylim(yrange)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            yerr=data[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        plots.append(p)
        labels.append(r"$L = %s$" % L)
        plotRange = FS3.getDataRange(data[L], idx=0, margin=0.05)
        g = np.linspace(plotRange[0], plotRange[1], 400)
        ax0.plot(g, fitFunction.func(g, L, params), **FS3.plotStyle(L))
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.78, 0.02, 0.18))
    ax0.set_title(r"$\alpha = %.1f$" % parameter_dict["alpha"][dict_index])
    ax0.axhline(var["a"][0], color="#dddddd", zorder=-1000)
    ax0.axvline(var["gc"], color="#dddddd", zorder=-1000)

    # inset (data collapse)
    fig, ax1 = FS3.addInset(fig, loc=3)
    ax1.set_xlabel(xlabel_red)
    if high_FS_effect:
        ax1.set_ylabel(ylabel_red)
        dataCollapse = FS3.rescaleAxis(
            data,
            xfunc="(x-gc)*L**(1./nu)",
            yfunc="y/(1+c*L**(-omega))",
            arg={
                "nu": var["nu"],
                "gc": var["gc"],
                "omega": var["omega"],
                "c": var["c"],
            },
        )
    else:
        ax1.set_ylabel(r"$U_2$")
        dataCollapse = FS3.rescaleAxis(
            data,
            xfunc="(x-gc)*L**(1./nu)",
            yfunc="y",
            arg={"nu": var["nu"], "gc": var["gc"]},
        )
    for L in np.sort(list(data), axis=0):
        ax1.errorbar(
            dataCollapse[L][:, 0],
            dataCollapse[L][:, 1],
            yerr=dataCollapse[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
    ax1.axhline(var["a"][0], color="#dddddd")
    ax1.axvline(0, color="#dddddd")

    if save:
        fig.savefig(
            dir + "Binder_alpha%.1f_zoom.pdf" % parameter_dict["alpha"][dict_index]
        )
        f = open(
            dir + "Binder_alpha%.1f_fit.txt" % parameter_dict["alpha"][dict_index], "w"
        )
        f.write(output)
        f.close()

    return var


def Binder_Hist(
    parameter_dict,
    dict_index,
    polyOrder=2,
    high_FS_effect=True,
    nsample=100,
    seed0=np.random.get_state(),
    save=False,
):
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    g0 = parameter_dict["gc"][dict_index]
    nu0 = parameter_dict["nu"][dict_index]

    np.random.set_state(seed0)

    # Histogram
    resList = np.empty((0, 5))
    aList = np.empty((0, polyOrder+1))
    for i in range(nsample):
        if high_FS_effect:
            fitFunction = f1()
        else:
            fitFunction = f0()
        fitFunction.polyOrder = polyOrder

        params0 = np.hstack(
            (
                [g0 + np.random.randn() * 0.01, nu0 + np.random.randn() * 0.1],
                np.random.randn(fitFunction.nparams() - 2),
            )
        )
        res = FS3.fit_minimize(
            resample(data), fitFunction, params=params0, **{"method": "BFGS"}
        )
        params, dparams, redChi2, ierr, msg = res
        if (
            np.abs(fitFunction.unpack(params)["gc"] - g0) < 0.5
            and np.abs(fitFunction.unpack(params)["nu"] - nu0) < 1.0
        ):
            if high_FS_effect:
                var = np.array(
                    [
                        fitFunction.unpack(params)["gc"],
                        fitFunction.unpack(params)["nu"],
                        fitFunction.unpack(params)["c"],
                        fitFunction.unpack(params)["omega"],
                        redChi2,
                    ]
                )
            else:
                var = np.array(
                    [
                        fitFunction.unpack(params)["gc"],
                        fitFunction.unpack(params)["nu"],
                        redChi2,
                    ]
                )
            resList = np.vstack((resList, var))
            aList = np.vstack((aList,fitFunction.unpack(params)["a"]))

    print("gc = %s" % (FS3.errText(np.mean(resList[:, 0]), np.std(resList[:, 0]))))
    print("nu = %s" % (FS3.errText(np.mean(resList[:, 1]), np.std(resList[:, 1]))))
    print(len(resList))

    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.hist(resList[:, 0], np.linspace(0.99 * g0, 1.01 * g0, 100))
    ax0.set_xlabel(r"$g_c$")
    ax1 = fig.add_subplot(132)
    ax1.hist(resList[:, 1], np.linspace(nu0 - 0.1, nu0 + 0.1, 100))
    ax1.set_xlabel(r"$\nu$")
    ax2 = fig.add_subplot(133)
    ax2.hist(
        resList[:, 2], np.linspace(0, 1.5 * parameter_dict["chiMax"][dict_index], 100)
    )
    ax2.set_xlabel(r"$\chi^2$")

    return resList,aList


def BinderT_Hist_fixed_nu(
    parameter_dict,
    dict_index,
    polyOrder=2,
    nsample=100,
    nu0=1.0,
    seed0=np.random.get_state(),
    save=False,
):
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    g0 = parameter_dict["gc"][dict_index]

    np.random.set_state(seed0)

    # Histogram
    resList = np.empty((0, 2))
    for i in range(nsample):
        fitFunction = f_fixed_nu(nu=nu0)
        fitFunction.polyOrder = polyOrder

        params0 = np.hstack(
            (
                [g0 + np.random.randn() * 0.01],
                np.random.randn(fitFunction.nparams() - 1),
            )
        )
        res = FS3.fit_minimize(
            resample(data), fitFunction, params=params0, **{"method": "BFGS"}
        )
        params, dparams, redChi2, mesg, ierr = res
        if np.abs(fitFunction.unpack(params)["gc"] - g0) < 0.5:
            resList = np.vstack(
                (resList, np.array([fitFunction.unpack(params)["gc"], redChi2]))
            )
    print("gc = %s" % (FS3.errText(np.mean(resList[:, 0]), np.std(resList[:, 0]))))
    print(len(resList))

    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.hist(resList[:, 0], np.linspace(0.99 * g0, 1.01 * g0, 100))
    ax0.set_xlabel(r"$g_c$")
    ax1 = fig.add_subplot(132)
    ax1.hist(resList[:, 1], np.linspace(nu0 - 0.1, nu0 + 0.1, 100))
    ax1.set_xlabel(r"$\nu$")
    ax2 = fig.add_subplot(133)
    ax2.hist(
        resList[:, 2], np.linspace(0, 1.5 * parameter_dict["chiMax"][dict_index], 100)
    )
    ax2.set_xlabel(r"$\chi^2$")

    return resList


def plot_Spin_stiffness(
    parameter_dict,
    dict_index,
    polyOrder=3,
    high_FS_effect=True,
    seed0=np.random.get_state(),
    save=False,
    finite_T=True,
    L_power=0.0,
):
    # raw plot
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "spin_stiffness%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    if finite_T:
        xlabel = r"$T$"
        xlabel_red = r"$(T-T_c)\, L^{1/\nu}$"
        dir = "plotsT/"
    else:
        xlabel = r"$J^{\perp}$"
        xlabel_red = r"$(J^{\perp}-J^{\perp}_c)\, L^{1/\nu}$"
        dir = "plotsJ/"

    ylabel_red = r"$\rho_s\,L^{%.1f}/(1+cL^{-\omega})$" % (L_power)

    fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=r"$\rho_s\,L^{%.1f}$" % (L_power))
    xrange = FS3.getDataRange(data, idx=0, margin=0.05)

    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        factor = float(L) ** (L_power)
        data[L][:, 1] *= factor
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            yerr=data[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        ax0.plot(data[L][:, 0], data[L][:, 1], **FS3.plotStyle(L), linestyle="-")
        plots.append(p)
        labels.append(r"$L = %s$" % L)

    ax0.set_xlim(data[128][0, 0], data[128][-1, 0])
    ax0.set_ylim(data[128][-1, 1], data[128][0, 1])
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.74, 0.72, 0.18))
    ax0.set_title(r"$\alpha = %.1f$" % parameter_dict["alpha"][dict_index])
    if save:
        fig.savefig(
            dir + "Spin_stiffness_alpha%.1f.pdf" % parameter_dict["alpha"][dict_index]
        )

    # fitted plot
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    if high_FS_effect:
        fitFunction = f1()
    else:
        fitFunction = f0()
    fitFunction.polyOrder = polyOrder

    g0 = parameter_dict["gc"][dict_index]
    nu0 = parameter_dict["nu"][dict_index]

    np.random.set_state(seed0)
    while True:

        seed = np.random.get_state()
        params0 = np.hstack(
            (
                [g0 + np.random.randn() * 0.01, nu0 + np.random.randn() * 0.01],
                0.01 * np.random.randn(fitFunction.nparams() - 2),
            )
        )
        res = FS3.fit_minimize(data, fitFunction, params=params0, **{"method": "BFGS"})
        params, dparams, redChi2, mesg, ierr = res

        if redChi2 < parameter_dict["chiMax"][dict_index]:
            break

    output = FS3.fitSummary(fitFunction, res)

    # main panel, crossings
    var = fitFunction.unpack(params)
    fig, ax0 = FS3.figure(xlabel=xlabel, ylabel=r"$\rho_s\,L^{%.1f}$" % (L_power))
    xrange, yrange = FS3.getDataRange(data, idx=0, margin=0.05), FS3.getDataRange(
        data, idx=1, margin=0.05
    )
    yrange[1] = 1.05 * yrange[1]
    ax0.set_xlim(xrange)
    ax0.set_ylim(yrange)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 1],
            yerr=data[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        plots.append(p)
        labels.append(r"$L = %s$" % L)
        plotRange = FS3.getDataRange(data[L], idx=0, margin=0.05)
        g = np.linspace(plotRange[0], plotRange[1], 400)
        ax0.plot(g, fitFunction.func(g, L, params), **FS3.plotStyle(L))
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.78, 0.02, 0.18))
    ax0.axhline(var["a"][0], color="#dddddd", zorder=-1000)
    ax0.axvline(var["gc"], color="#dddddd", zorder=-1000)

    # inset (data collapse)
    fig, ax1 = FS3.addInset(fig, loc=3)
    ax1.set_xlabel(xlabel_red)
    if high_FS_effect:
        ax1.set_ylabel(ylabel_red)
        dataCollapse = FS3.rescaleAxis(
            data,
            xfunc="(x-gc)*L**(1./nu)",
            yfunc="y/(1+c*L**(-omega))",
            arg={
                "nu": var["nu"],
                "gc": var["gc"],
                "omega": var["omega"],
                "c": var["c"],
            },
        )
    else:
        ax1.set_ylabel(r"$\rho_s\,L^{%.1f}$" % (L_power))
        dataCollapse = FS3.rescaleAxis(
            data,
            xfunc="(x-gc)*L**(1./nu)",
            yfunc="y",
            arg={"nu": var["nu"], "gc": var["gc"]},
        )
    for L in np.sort(list(data), axis=0):
        ax1.errorbar(
            dataCollapse[L][:, 0],
            dataCollapse[L][:, 1]/(1+var["c"]*L**(-var["omega"])),
            yerr=dataCollapse[L][:, 2]/(1+var["c"]*L**(-var["omega"])),
            **FS3.errorbarStyle(L),
            linestyle=""
        )
    ax1.axhline(var["a"][0], color="#dddddd")
    ax1.axvline(0, color="#dddddd")

    if save:
        fig.savefig(
            dir
            + "Spin_stiffness_alpha%.1f_zoom.pdf" % parameter_dict["alpha"][dict_index]
        )
        f = open(
            dir
            + "Spin_stiffness_alpha%.1f_fit.txt" % parameter_dict["alpha"][dict_index],
            "w",
        )
        f.write(output)
        f.close()

    return var


def Spin_Stiffness_Hist(
    parameter_dict,
    dict_index,
    polyOrder=2,
    high_FS_effect=True,
    nsample=100,
    seed0=np.random.get_state(),
    save=False,
    L_power=0.0,
):
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "spin_stiffness%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    for L in np.sort(list(data), axis=0):
        factor = float(L) ** (L_power)
        data[L][:, 1] *= factor

    g0 = parameter_dict["gc"][dict_index]
    nu0 = parameter_dict["nu"][dict_index]

    np.random.set_state(seed0)

    # Histogram
    resList = np.empty((0, 3))
    for i in range(nsample):
        if high_FS_effect:
            fitFunction = f1()
        else:
            fitFunction = f0()
        fitFunction.polyOrder = polyOrder

        params0 = np.hstack(
            (
                [g0 + np.random.randn() * 0.01, nu0 + np.random.randn() * 0.1],
                np.random.randn(fitFunction.nparams() - 2),
            )
        )
        res = FS3.fit_minimize(
            resample(data), fitFunction, params=params0, **{"method": "BFGS"}
        )
        params, dparams, redChi2, ierr, msg = res
        if (
            np.abs(fitFunction.unpack(params)["gc"] - g0) < 0.5
            and np.abs(fitFunction.unpack(params)["nu"] - nu0) < 1.0
        ):
            resList = np.vstack(
                (
                    resList,
                    np.array(
                        [
                            fitFunction.unpack(params)["gc"],
                            fitFunction.unpack(params)["nu"],
                            redChi2,
                        ]
                    ),
                )
            )
    print("gc = %s" % (FS3.errText(np.mean(resList[:, 0]), np.std(resList[:, 0]))))
    print("nu = %s" % (FS3.errText(np.mean(resList[:, 1]), np.std(resList[:, 1]))))
    print(len(resList))

    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.hist(resList[:, 0], np.linspace(0.99 * g0, 1.01 * g0, 100))
    ax0.set_xlabel(r"$g_c$")
    ax1 = fig.add_subplot(132)
    ax1.hist(resList[:, 1], np.linspace(nu0 - 0.1, nu0 + 0.1, 100))
    ax1.set_xlabel(r"$\nu$")
    ax2 = fig.add_subplot(133)
    ax2.hist(
        resList[:, 2], np.linspace(0, 1.5 * parameter_dict["chiMax"][dict_index], 100)
    )
    ax2.set_xlabel(r"$\chi^2$")

    return resList


def mag2_U(
    parameter_dict,
    dict_index,
    polyOrder=2,
    high_FS_effect=True,
    seed0=np.random.get_state(),
    save=False,
):
    dataM = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    dataU = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )

    Llist = np.sort(list(dataM), axis=0)
    FS3.setStyle(Llist=Llist)

    eta0 = parameter_dict["eta"][dict_index]

    fig, ax0 = FS3.figure(xlabel=r"$U_2$", ylabel=r"$m^2$")
    xrange = FS3.getDataRange(dataU, idx=1, margin=0.05)
    yrange = FS3.getDataRange(dataM, idx=1, margin=0.05)

    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []

    for L in Llist:
        p = ax0.errorbar(
            dataU[L][:, 1],
            dataM[L][:, 1],
            yerr=dataM[L][:, 2],
            xerr=dataU[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        ax0.plot(dataU[L][:, 1], dataM[L][:, 1], **FS3.plotStyle(L), linestyle="-")
        plots.append(p)
        labels.append(r"$L = %s$" % L)

    ax0.set_xlim(xrange)
    ax0.set_ylim(yrange)
    # ax0.set_xlim(dataU[128][-1, 1], dataU[128][0, 1])
    # ax0.set_ylim(dataM[128][-1, 1], dataM[128][0, 1])
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.74, 0.72, 0.18))
    ax0.set_title(r"$\alpha = %.1f$" % parameter_dict["alpha"][dict_index])

    # fitted plot
    dataU = FS3.select(
        dataU,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    dataM = FS3.select(
        dataM,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    FS3.setStyle(Llist=np.sort(list(dataM), axis=0))

    Llist = np.sort(list(dataM), axis=0)

    data = {}
    for L in Llist:
        Tlist = dataU[L][:, 0]
        data[L] = np.empty((0, 4))
        for T in Tlist:
            iU = np.argwhere(dataU[L][:, 0] == T)[0, 0]
            iM = np.argwhere(dataM[L][:, 0] == T)[0, 0]
            data[L] = np.vstack(
                (
                    data[L],
                    np.array(
                        [
                            dataU[L][iU, 1],
                            dataU[L][iU, 2],
                            dataM[L][iM, 1],
                            dataM[L][iM, 2],
                        ]
                    ),
                )
            )

    if high_FS_effect:
        fitFunction = Rg1()
        xlabel_red = r"$U_2/(1+cL^{\omega})$"
        ylabel_red = r"$m^2L^{\eta}/(1+c^{\prime}L^{\omega^{\prime}})$"
    else:
        fitFunction = Rg0()
        xlabel_red = r"$U_2$"
        ylabel_red = r"$m^2L^{\eta}$"
    fitFunction.polyOrder = polyOrder

    np.random.set_state(seed0)
    while True:

        seed = np.random.get_state()
        params0 = np.hstack(
            (
                [eta0 + np.random.randn() * 0.01],
                0.01 * np.random.randn(fitFunction.nparams() - 1),
            )
        )
        res = FS3.fit_minimize(data, fitFunction, params=params0, **{"method": "BFGS"})
        params, dparams, redChi2, mesg, ierr = res

        if redChi2 < parameter_dict["chiMax"][dict_index]:
            break

    print("redChi2:", redChi2)
    output = FS3.fitSummary(fitFunction, res)

    # main panel, crossings
    var = fitFunction.unpack(params)
    dvar = fitFunction.unpack(dparams)
    print(var)
    print(dvar)
    fig, ax0 = FS3.figure(xlabel=r"$U_2$", ylabel=r"$m^2$")
    xrange, yrange = FS3.getDataRange(data, idx=0, margin=0.05), FS3.getDataRange(
        data, idx=2, margin=0.05
    )
    yrange[1] = 1.05 * yrange[1]
    # ax0.set_xlim(xrange)
    ax0.set_ylim(yrange)
    FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    plots = []
    labels = []
    for L in np.sort(list(data), axis=0):
        p = ax0.errorbar(
            data[L][:, 0],
            data[L][:, 2],
            xerr=data[L][:, 1],
            yerr=data[L][:, 3],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
        plots.append(p)
        labels.append(r"$L = %s$" % L)
        plotRange = FS3.getDataRange(data[L], idx=0, margin=0.05)
        g = np.linspace(plotRange[0], plotRange[1], 400)
        ax0.plot(g, fitFunction.func(g, L, params), **FS3.plotStyle(L))
    FS3.legend(ax0, plots, labels, legendLabelBreak=7, loc=(0.78, 0.02, 0.18))
    ax0.set_title(r"$\alpha = %.1f$" % (parameter_dict["alpha"][dict_index]))

    # inset (data collapse)
    fig, ax1 = FS3.addInset(fig, loc=3)
    ax1.set_xlabel(xlabel_red)
    ax1.set_ylabel(ylabel_red)

    for L in np.sort(list(data), axis=0):
        if high_FS_effect:
            factor = 1.0 / (1 + var["c"] * L ** (-var["omega"]))
            factorP = 1.0 / (1 + var["cPrime"] * L ** (-var["omegaPrime"]))
        else:
            factor = 1.0
        ax1.errorbar(
            data[L][:, 0] * factor,
            data[L][:, 2] * L ** (var["eta"])*factorP,
            xerr=data[L][:, 1],
            yerr=data[L][:, 3] * L ** (var["eta"])*factorP,
            **FS3.errorbarStyle(L),
            linestyle=""
        )


def Eta_Hist(
    parameter_dict,
    dict_index,
    polyOrder=2,
    high_FS_effect=True,
    nsample=100,
    seed0=np.random.get_state(),
    save=False,
):
    dataM = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    dataU = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    dataM = FS3.select(
        dataM,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    dataU = FS3.select(
        dataU,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )

    Llist = np.sort(list(dataM), axis=0)

    data = {}
    for L in Llist:
        Tlist = dataU[L][:, 0]
        data[L] = np.empty((0, 4))
        for T in Tlist:
            iU = np.argwhere(dataU[L][:, 0] == T)[0, 0]
            iM = np.argwhere(dataM[L][:, 0] == T)[0, 0]
            data[L] = np.vstack(
                (
                    data[L],
                    np.array(
                        [
                            dataU[L][iU, 1],
                            dataU[L][iU, 2],
                            dataM[L][iM, 1],
                            dataM[L][iM, 2],
                        ]
                    ),
                )
            )
    FS3.setStyle(Llist=Llist)

    eta0 = parameter_dict["eta"][dict_index]

    np.random.set_state(seed0)

    # Histogram
    if high_FS_effect:
        fitFunction = Rg1()
        resList = np.empty((0, 6))
    else:
        fitFunction = Rg0()
        resList = np.empty((0, 2))
    fitFunction.polyOrder = polyOrder
    aList = np.empty((0, polyOrder + 1))
    errList = np.copy(resList)

    for i in range(nsample):
        params0 = np.hstack(
            (
                [
                    eta0 + np.random.randn() * 0.01,
                    0.0 + np.random.randn(),
                    1 + np.random.randn(),
                ],
                np.random.randn(fitFunction.nparams() - 3),
            )
        )
        res = FS3.fit_minimize(
            data, fitFunction, params=params0, **{"method": "BFGS"}
        )
        params, dparams, redChi2, ierr, msg = res
        if np.abs(fitFunction.unpack(params)["eta"] - eta0) < 0.5:
            if high_FS_effect:
                var = np.array(
                    [
                        fitFunction.unpack(params)["eta"],
                        fitFunction.unpack(params)["c"],
                        fitFunction.unpack(params)["omega"],
                        fitFunction.unpack(params)["cPrime"],
                        fitFunction.unpack(params)["omegaPrime"],
                        redChi2,
                    ]
                )
                dvar = np.array(
                    [
                        fitFunction.unpack(dparams)["eta"],
                        fitFunction.unpack(dparams)["c"],
                        fitFunction.unpack(dparams)["omega"],
                        fitFunction.unpack(params)["cPrime"],
                        fitFunction.unpack(params)["omegaPrime"],
                        redChi2,
                    ]
                )
            else:
                var = np.array(
                    [
                        fitFunction.unpack(params)["eta"],
                        redChi2,
                    ]
                )
                dvar = np.array(
                    [
                        fitFunction.unpack(dparams)["eta"],
                        redChi2,
                    ]
                )
            resList = np.vstack((resList,var))
            errList = np.vstack((errList,dvar))
            aList = np.vstack((aList, fitFunction.unpack(params)["a"]))

    print(
        "eta (hist) = %s" % (FS3.errText(np.mean(resList[:, 0]), np.std(resList[:, 0])))
    )
    print(len(resList))

    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    ax0.hist(resList[:, 0], np.linspace(0.9 * eta0, 1.1 * eta0, 10))
    ax0.set_xlabel(r"$\eta$")
    ax2 = fig.add_subplot(122)
    ax2.hist(
        resList[:, 1], np.linspace(0, 1.5 * parameter_dict["chiMax"][dict_index], 10)
    )
    ax2.set_xlabel(r"$\chi^2$")

    return resList, errList, aList

def data_collapse(parameter_dict, dict_index,gc,nu,eta):
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    Llist = np.sort(list(data), axis=0)
    FS3.setStyle(Llist=Llist)
    fig, ax0 = FS3.figure(xlabel=r"$(g-g_c)L^{1/\nu}$", ylabel=r"$m^2L^{\eta}$")
    for L in Llist:
        ax0.errorbar(
            (data[L][:, 0]-gc)*L**(1./nu),
            data[L][:, 1]*L**eta,
            yerr=data[L][:, 2]*L**eta,
            **FS3.errorbarStyle(L),
            linestyle="",
            label=r"$L = %d$" % L
        )
        g = np.linspace(
            FS3.getDataRange(data[L])[0] * 0.95,
            FS3.getDataRange(data[L])[1] * 1.05,
            1000,
        )
    ax0.legend()

def mag2_L(parameter_dict, dict_index, save=False):
    data = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )

    data = FS3.select(
        data,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    Llist = np.sort(list(data), axis=0)
    FS3.setStyle(Llist=Llist)

    g0 = parameter_dict["gc"][dict_index]
    nu0 = parameter_dict["nu"][dict_index]

    tab = np.empty((0, 3))
    for L in np.sort(list(data.keys()), axis=0):
        imin = np.argmin(np.abs(data[L][:, 0] - g0))
        tab = np.vstack((tab, np.array([L, data[L][imin, 1], data[L][imin, 2]])))
    print("gc = %f" % (g0))
    print("@ g = %f" % (data[L][imin, 0]))

    fig, ax0 = FS3.figure(xlabel=r"$L$", ylabel=r"$m^2$")
    ax0.errorbar(
        tab[:, 0], tab[:, 1], yerr=tab[:, 2], **FS3.errorbarStyle(32), linestyle=""
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    # ax0.set_xticks(Llist)
    FS3.setTicks(ax0, xTicks=Llist)
    for axis in [ax0.xaxis]:
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax0.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    mean, err, rchi2 = fit(tab[:, 0], tab[:, 1], yerr=tab[:, 2], p0=[0.1, 0.7])
    x = np.linspace(30, 260, 1000)
    ax0.plot(x, powerlaw(mean, x), "-", **FS3.plotStyle(32))
    print("eta (m^2(L)) = %s" % (FS3.errText(mean[1], err[1])))
    if save:
        fig.savefig("mag2_L_alpha%s.pdf", parameter_dict["alpha"][dict_index])


def datacollapse(parameter_dict, dict_index, polyOrder=3, save=False):
    dataR = FS3.load(
        parameter_dict["path_to_data"]
        + "Binder%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    dataM = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    dataR = FS3.select(
        dataR,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    dataM = FS3.select(
        dataM,
        gRange=parameter_dict["gRange"][dict_index],
        LRange=parameter_dict["LRange"][dict_index],
    )
    Llist = np.sort(list(dataR), axis=0)

    data = {}
    for L in Llist:
        Tlist = dataR[L][:, 0]
        data[L] = np.empty((0, 4))
        for T in Tlist:
            iR = np.argwhere(dataR[L][:, 0] == T)[0, 0]
            iM = np.argwhere(dataM[L][:, 0] == T)[0, 0]
            data[L] = np.vstack(
                (
                    data[L],
                    np.array(
                        [
                            dataR[L][iR, 1],
                            dataR[L][iR, 2],
                            dataM[L][iM, 1],
                            dataM[L][iM, 2],
                        ]
                    ),
                )
            )

    fitFunction = Rg0()
    fitFunction.polyOrder = polyOrder

    params0 = np.zeros(fitFunction.polyOrder + 1)
    params0[0] = 0.982
    params0[1:] = np.random.randn(fitFunction.polyOrder)
    res = FS3.fit_minimize(data, fitFunction, params=params0, **{"method": "BFGS"})
    params, dparams, redChi2, mesg, ierr = res

    FS3.fitSummary(fitFunction, res)

    FS3.setStyle(Llist=Llist)
    fig, ax0 = FS3.figure(xlabel=r"$U^2$", ylabel=r"$m^2$")
    for L in Llist:
        ax0.errorbar(
            data[L][:, 0],
            data[L][:, 2],
            xerr=data[L][:, 1],
            yerr=data[L][:, 3],
            **FS3.errorbarStyle(L),
            linestyle="",
            label=r"$L = %d$" % L
        )
        g = np.linspace(
            FS3.getDataRange(data[L])[0] * 0.95,
            FS3.getDataRange(data[L])[1] * 1.05,
            1000,
        )
        ax0.plot(g, fitFunction.func(g, L, params), **FS3.plotStyle(L))
    ax0.legend()

    eta = fitFunction.unpack(params)["eta"]
    nu = parameter_dict["nu"][dict_index]
    # c = var['c']
    # omega = var['omega']
    gc = parameter_dict["gc"][dict_index]

    ## OWN IMPLEMENTATION!!!!!!!!!
    # inset (data collapse)
    fig, ax1 = FS3.addInset(fig, loc=3)
    ax1.set_xlabel(r"$U_2$")
    ax1.set_ylabel(r"$m^2\, L^{\eta}$")
    dataCollapse = FS3.rescaleAxis(
        data,
        xfunc="x",
        yfunc="y*L**(eta)",
        arg={"eta": eta},
    )
    for L in np.sort(list(data), axis=0):
        ax1.errorbar(
            dataCollapse[L][:, 0],
            dataCollapse[L][:, 1],
            yerr=dataCollapse[L][:, 2],
            **FS3.errorbarStyle(L),
            linestyle=""
        )
    # ax1.axhline(var["a"][0], color="#dddddd")
    # ax1.axvline(0, color="#dddddd")

    data = FS3.load(
        parameter_dict["path_to_data"]
        + "mag2%s.dat" % (parameter_dict["sigma"][dict_index]),
        LIndex=0,
        xIndex=1,
        yIndex=2,
        dyIndex=3,
    )
    FS3.setStyle(Llist=np.sort(list(data), axis=0))

    # fig, ax0 = FS3.figure(xlabel=r'$(g-gc) L^{1/\nu}(1+cL^{\omega})$', ylabel=r'$m^2 L^{2\beta/\nu}$')
    fig, ax0 = FS3.figure(xlabel=r"$(g-gc) L^{1/\nu}$", ylabel=r"$m^2 L^{2\beta/\nu}$")
    # ax0.set_xlim(-9,5.5)
    # ax0.set_ylim(0.,1)
    # FS3.setTicks(ax0, minorxTicks=2, minoryTicks=2)
    for L in np.sort(list(data.keys()), axis=0):
        # ax0.errorbar((data[L][:,0]-gc)*L**(1/nu)*(1 + c*L**(omega/nu)), data[L][:,1]*L**(1-eta), yerr=data[L][:,2]*L**eta, **FS3.errorbarStyle(L), linestyle='', label=r'$L=%d$'%L)
        ax0.errorbar(
            (data[L][:, 0] - gc) * L ** (1 / nu),
            data[L][:, 1] * L ** (eta),
            yerr=data[L][:, 2] * L**eta,
            **FS3.errorbarStyle(L),
            linestyle="",
            label=r"$L=%d$" % L
        )
    ax0.legend(loc=(0.71, 0.72))
    if save:
        fig.savefig("Data_collapse_alpha%s.pdf", parameter_dict["alpha"][dict_index])
"""
