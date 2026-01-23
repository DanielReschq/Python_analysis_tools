# 	FS3.py -- a Python module to simplify the finite scaling analysis of simulation data
# 	by Thomas C. Lang <thomas.lang@uibk.ac.at>
#  03.05.2024

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import sympy
import copy
import errno
import os
import sys

__version__ = "1.0.0"

_style = {}
_style["figure.labelsize"] = mpl.rcParams["xtick.labelsize"]
# _style['axes.titlesizeInset'] = 0.89 * mpl.rcParams['axes.titlesize']
# _style['axes.labelsizeInset'] = 0.89 * mpl.rcParams['axes.labelsize']
# _style['xtick.labelsizeInset'] = 0.89 * mpl.rcParams['xtick.labelsize']
# _style['ytick.labelsizeInset'] = 0.89 * mpl.rcParams['ytick.labelsize']
# _style['legend.fontsizeInset'] = 0.89 * mpl.rcParams['legend.fontsize']
# _style['figure.labelsizeInset'] = 0.89 * mpl.rcParams['xtick.labelsize']
_style["markerEdgeWidth"] = 1.6
_style["capSize"] = 3
_style["lineWidth"] = 1.2
_style["marker"] = np.array(
    [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
        "o",
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
        "o",
    ]
)
_style["markerSize"] = np.array(
    [
        6.5,
        5.25,
        5.75,
        6.75,
        6.75,
        6.75,
        6.75,
        7.1,
        6.9,
        6.5,
        6.5,
        5.25,
        5.75,
        6.75,
        6.75,
        6.75,
        6.75,
        7.1,
        6.9,
        6.5,
    ]
)
_style["markerEdgeColor"] = np.array(
    [
        "#b11823",
        "#dc6707",
        "#efc600",
        "#5dc661",
        "#60af92",
        "#505ea5",
        "#25305f",
        "#aa1966",
        "#231f20",
        "#000000",
        "#b11823",
        "#dc6707",
        "#efc600",
        "#5dc661",
        "#60af92",
        "#505ea5",
        "#25305f",
        "#aa1966",
        "#231f20",
        "#000000",
    ]
)
_style["markerFaceColor"] = np.array(
    [
        "#cf9494",
        "#e6b594",
        "#f3e2a0",
        "#bbe0b5",
        "#b7d5c9",
        "#a8afce",
        "#8594bd",
        "#cd95b1",
        "#908e8f",
        "#666666",
        "#cf9494",
        "#e6b594",
        "#f3e2a0",
        "#bbe0b5",
        "#b7d5c9",
        "#a8afce",
        "#8594bd",
        "#cd95b1",
        "#908e8f",
        "#666666",
    ]
)
_style["Ldict"] = {}


# -----------------------------------------------------------------------------------------
def fit_residuals(params, sum, data, fitFunction):

    res = []
    for L in list(data):
        x, y = data[L][:, 0], data[L][:, 1]
        if len(data[L][0]) == 2:  # no errors
            r = y - fitFunction.func(x, L, params)
        elif len(data[L][0]) == 3:  # errors for y only
            dy = data[L][:, 2]
            r = (y - fitFunction.func(x, L, params)) / dy
        elif len(data[L][0]) == 4:  # errors for x and y
            dx, dy = data[L][:, 2], data[L][:, 3]
            r = (y - fitFunction.func(x, L, params)) / np.sqrt(dx**2 + dy**2)
        else:
            sys.exit("FS3.fit_residuals: Incorrect data structure.")

        res.extend(r)

    if sum:
        return np.sum(np.array(res)**2)
    else:
        return np.array(res)


# -----------------------------------------------------------------------------------------
def fit_leastsq(data, fitFunction, params=[]):

    if len(params) == 0:
        params = np.random.random(fitFunction.nparams())

    params, dparams2, info, mesg, ierr = sp.optimize.leastsq(
        fit_residuals,
        params,
        args=(False, data, fitFunction),
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        maxfev=10**6,
        full_output=True,
    )

    if ierr < 5:
        redChi2 = fit_residuals(params, True, data, fitFunction) / (
            np.sum(np.fromiter((len(data[L]) for L in list(data)), np.double))
            - fitFunction.nparams()
        )
        if dparams2 is not None:
            dparams = np.sqrt(np.diag(dparams2))
        else:
            dparams = None
    else:
        dparams = np.ones_like(params) * float("nan")
        redChi2 = -1

    return params, dparams, redChi2, ierr, mesg


# -----------------------------------------------------------------------------------------
def fit_minimize(data, fitFunction, params=[], **kwargs):

    if len(params) == 0:
        params = np.random.random(fitFunction.nparams())

    result = sp.optimize.minimize(
        fit_residuals, params, args=(True, data, fitFunction), **kwargs
    )
    params, ierr, mesg = result.x, result.status, result.message

    if ierr < 5:
        redChi2 = fit_residuals(params, True, data, fitFunction) / (
            np.sum(np.fromiter((len(data[L]) for L in list(data)), np.double))
            - fitFunction.nparams()
        )
        dparams = np.sqrt(np.diag(result.hess_inv))
    else:
        redChi2 = -1
        dparams = np.ones_like(params) * float("nan")

    return params, dparams, redChi2, ierr, mesg


# -----------------------------------------------------------------------------------------
def fit_precond(data, fitFunction, params=[]):

    if len(params) == 0:
        params = np.random.random(fitFunction.nparams())

    result = sp.optimize.minimize(
        fit_residuals,
        params,
        args=(True, data, fitFunction),
        method="BFGS",
        tol=1.0e-8,
        options={"gtol": 1e-8},
    )
    params = result.x

    params, dparams, redChi2, ierr, mesg = fit_leastsq(data, fitFunction, params=params)

    return params, dparams, redChi2, ierr, mesg


# -----------------------------------------------------------------------------------------
def fitSummary(fitFunction, res, silent=False):

    params, dparams, redChi2, ierr, mesg = res
    var = fitFunction.unpack(params)
    dvar = fitFunction.unpack(dparams)

    output = ""
    if ierr < 5:
        #        output += 'Fit successful (%d): %s\n'%(ierr, mesg)
        output += "chi2/d.o.f = {:<14.2f}\n".format(redChi2)
        for name in list(var):
            if isinstance(var[name], np.ndarray):
                for i in range(len(var[name])):
                    output += "%s[%d] = %s\n" % (
                        name,
                        i,
                        errText(var[name][i], dvar[name][i]),
                    )
            else:
                output += "%s = %s\n" % (name, errText(var[name], dvar[name]))
    else:
        output += "Fit failed (%d): %s\n" % (ierr, mesg)

    if not silent:
        print("%s" % output)

    return output


# -----------------------------------------------------------------------------------------
def load(file, LIndex=0, xIndex=1, yIndex=2, dyIndex=3, dxIndex=None):
    """Load finite size data from file into a dictionary, where the keys correspond to
    the available system sizes. The expected data structure is, e.g.,

    #0
    5     6.50     0.32276249    0.00063628
    5     7.50     0.40757436    0.00054231
    5     8.50     0.50600748    0.00017793

    #1
    7     6.50     0.26643471    0.00064648
    7     7.50     0.36700643    0.00072537
    7     8.50     0.50036142    0.00067841
    ...
    """

    if not os.path.isfile(file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    if dxIndex != None:
        cols = (LIndex, xIndex, dxIndex, yIndex, dyIndex)
    else:
        cols = (LIndex, xIndex, yIndex, dyIndex)
    tab = np.loadtxt(file, dtype=float, comments="#", usecols=cols)

    L0 = int(tab[0, LIndex])
    data = {}
    dataL = np.empty((0, len(cols) - 1))

    for i in range(len(tab)):
        if int(tab[i, LIndex]) == L0:
            dataL = np.vstack((dataL, np.take(tab[i], [xIndex, yIndex, dyIndex])))
        if int(tab[i, LIndex]) != L0:
            data[L0] = dataL
            dataL = np.array([np.take(tab[i], [xIndex, yIndex, dyIndex])])
            L0 = int(tab[i, LIndex])
    data[L0] = dataL

    #   missing: dx

    return data


# -----------------------------------------------------------------------------------------
def select(dataIn, LRange=[], gRange=[]):
    """Select a subset of the data dictionary according to the provided criteria."""

    if len(LRange) == 0:
        LRange = np.array([-sys.maxsize - 1, sys.maxsize])
    elif len(LRange) == 1:
        LRange = np.array([LRange[0], sys.maxsize])
    else:
        LRange = np.array(LRange)

    Llist = np.array(list(dataIn))

    Llist = Llist[np.where((Llist >= LRange[0]) & (Llist <= LRange[1]))]

    if len(gRange) == 0:
        gRange = np.array([float("-inf"), float("inf")])
    elif len(gRange) == 1:
        gRange = np.array([gRange[0], float("inf")])
    else:
        gRange = np.array(gRange)

    dataOut = {}
    for L in Llist:
        dataOut[L] = dataIn[L][
            np.where((dataIn[L][:, 0] >= gRange[0]) & (dataIn[L][:, 0] <= gRange[1]))
        ]

    return dataOut


# -----------------------------------------------------------------------------------------
def getDataRange(data, idx=0, margin=0.0):

    dataMin, dataMax = float("inf"), float("-inf")

    if type(data) == dict:
        for L in list(data):
            dataMin = min(np.amin(data[L][:, idx]), dataMin)
            dataMax = max(np.amax(data[L][:, idx]), dataMax)
    else:
        if len(data.shape) == 1:
            dataMin = np.amin(data)
            dataMax = np.amax(data)
        elif len(data.shape) == 2:
            dataMin = np.amin(data[:, idx])
            dataMax = np.amax(data[:, idx])

    return np.array(
        [
            dataMin - margin * (dataMax - dataMin),
            dataMin + (1.0 + margin) * (dataMax - dataMin),
        ]
    )


# -----------------------------------------------------------------------------------------
def errorPropagation(func, arg={}):
    """Return the leading order Gaussian error propagation of the provided function"""

    func = sympy.sympify(func)
    err = 0
    for name in func.free_symbols:
        err += func.diff(name) ** 2 * sympy.sympify("d" + str(name)) ** 2
    return sympy.sqrt(err).subs(arg)


# -----------------------------------------------------------------------------------------
def rescaleAxis(
    dataIn, xfunc="x", yfunc="y", xIndex=0, yIndex=1, dxIndex=None, dyIndex=None, arg={}
):
    """Rescale the x and y data according to the provided functional expressions. If
    indices to error-values are provided, Gaussian error propagation is used to compute
    transformed errorbars."""

    dataOut = copy.deepcopy(dataIn)

    if xfunc != None:
        xfunc = sympy.sympify(xfunc)
        if dxIndex != None:
            dxfunc = errorPropagation(xfunc)

    if yfunc != None:
        yfunc = sympy.sympify(yfunc)
        if dyIndex != None:
            dyfunc = errorPropagation(yfunc)

    for L in list(dataOut):
        for i in range(len(dataOut[L])):
            dict = arg
            dict["L"], dict["x"], dict["y"] = (
                L,
                dataOut[L][i, xIndex],
                dataOut[L][i, yIndex],
            )
            dataOut[L][i, xIndex] = xfunc.subs(dict).evalf()
            dataOut[L][i, yIndex] = yfunc.subs(dict).evalf()
            if dxIndex != None:
                dict["dx"] = dataOut[L][i, dxIndex]
                dataOut[L][i, dxIndex] = dxfunc.subs(dict).evalf()
            if dyIndex != None:
                dict["dy"] = dataOut[L][i, dyIndex]
                dataOut[L][i, dyIndex] = dyfunc.subs(dict).evalf()

    return dataOut


# -----------------------------------------------------------------------------------------
def dict_L2g(dataIn):
    """Restructure a data dictionary with system sizes as keys in order to return the data dictionary with keys corresponding to the available couplings."""

    Llist = np.array(list(dataIn))

    glist = np.array([])
    for L in Llist:
        glist = np.append(glist, dataIn[L][:, 0])
    glist = np.unique(glist)

    dataOut = {}
    for g in glist:
        tmp = np.empty((0, 3))
        for L in Llist:
            sel = dataIn[L][np.where(dataIn[L][:, 0] == g)]
            if len(sel) > 0:
                tmp = np.vstack((tmp, np.hstack((L, sel[0][1:]))))
        dataOut[g] = tmp

    return dataOut


# -----------------------------------------------------------------------------------------
def errText(val, err, maxDigit=16):
    """Return value and the significant digit of the error as a string, i.e., errText(12.34567, 0.00692) = 12.346(7)."""

    if err == 0.0:
        return ("%." + str(maxDigit) + "f") % val
    if err >= 1.0:
        if np.abs(val) >= 1.0:
            return "%d(%d)" % (round(val), round(err))
        else:
            if round(err) != 0:
                return "%d(%d)" % (round(val), round(err))
            else:
                return "%d(%.1f)" % (round(val), err)
    for d in range(250):
        if int((err - int(err)) * 10.0**d) != 0:
            errOrder = d
            break

    return (
        ("%." + str(errOrder) + "f") % val
        + "("
        + str(int(round(err * 10**errOrder)))
        + ")"
    )


# -----------------------------------------------------------------------------------------
def poly(p, x):

    f = 0.0
    for i in range(len(p)):
        f += p[i] * x**i

    return f


# -----------------------------------------------------------------------------------------
def fitPoly(x, y, order, xerr=0, yerr=0):

    if type(xerr) == int and type(yerr) == int:
        err_func = lambda p, x, y, xerr, yerr: (y - poly(p, x))
    else:
        if type(xerr) == int:
            xerr = np.zeros_like(x)
        if type(yerr) == int:
            yerr = np.zeros_like(y)
        err_func = lambda p, x, y, xerr, yerr: (y - poly(p, x)) / np.sqrt(
            xerr**2 + yerr**2
        )

    p0 = np.random.random(order + 1)
    mean, cov, info, mesg, ierr = sp.optimize.leastsq(
        err_func, p0, args=(x, y, xerr, yerr), full_output=1
    )
    rchi2 = np.sum(err_func(mean, x, y, xerr, yerr) ** 2) / (len(x) - len(mean))

    return mean, np.sqrt(rchi2 * np.diag(cov)), rchi2


# -----------------------------------------------------------------------------------------
def intersect(p1, p2, x0=0.0, x1=1.0):
    """Return the intersection of the polynomials defined by the coefficients p1 and p2,
    starting the search at x0 and x1."""

    f = lambda x, p1, p2: poly(p1, x) - poly(p2, x)
    if np.sign(f(x0, p1, p2)) == np.sign(f(x1, p1, p2)):
        return False, np.zeros(2)
    else:
        x, r = sp.optimize.brentq(
            f, x0, x1, args=(p1, p2), xtol=1.0e-15, rtol=1.0e-15, full_output=True
        )
        return r.converged, np.array([x, poly(p1, x)])


# -----------------------------------------------------------------------------------------
def bootstrapIntersect(data1, data2, order, nsamples=1000):
    """Bootstrap the intersection points for error estimates."""

    f = lambda x, p1, p2: poly(p1, x) - poly(p2, x)
    x0 = min(np.amin(data1[:, 0]), np.amin(data2[:, 0]))
    x1 = min(np.amax(data1[:, 0]), np.amax(data2[:, 0]))
    resList = np.empty((0, 2))
    for i in range(nsamples):
        y1s = data1[:, 1] + np.random.randn(len(data1)) * data1[:, 2]
        y2s = data2[:, 1] + np.random.randn(len(data2)) * data2[:, 2]
        p1, dp1, chi2 = fitPoly(data1[:, 0], y1s, order)
        p2, dp2, chi2 = fitPoly(data2[:, 0], y2s, order)
        conv, res = intersect(p1, p2, x0, x1)
        if conv:
            resList = np.vstack((resList, res))

    return np.array(
        [
            np.mean(resList[:, 0]),
            np.sqrt(np.var(resList[:, 0])),
            np.mean(resList[:, 1]),
            np.sqrt(np.var(resList[:, 1])),
        ]
    ), len(resList)


# -----------------------------------------------------------------------------------------
def crossingPoints(data, order=2, dLfunc="L+2", nsamples=1000):
    """Determine the intersection points of polynomial fits of a given order to two
    system sizes related by dLfunc."""

    Llist = np.sort(list(data))
    Lset = np.empty((0, 2))
    xpoints = np.empty((0, 4))

    for i in range(len(Llist)):
        L1, L2 = Llist[i], eval(dLfunc.replace("L", str(Llist[i])))
        if np.isin(L1, Llist) & np.isin(L2, Llist):
            Lset = np.vstack((Lset, [L1, L2]))
            xpoints = np.vstack(
                (
                    xpoints,
                    bootstrapIntersect(data[L1], data[L2], order, nsamples=nsamples)[0],
                )
            )

    return Lset, xpoints


# -----------------------------------------------------------------------------------------
def setStyle(Llist=[], dict={}):
    """Pass a list of system sizes to assign style definitions, or a customized dictionary
    to overide the FS3 default style definitions."""

    if len(Llist) != 0:
        i = 0
        for L in list(Llist):
            _style["Ldict"][L] = i
            i += 1

    if len(dict) != 0:
        for key, value in dict.items():
            _style[key] = value

    if (len(Llist) == 0) and (len(dict) == 0):
        for i in range(10):
            _style["Ldict"][i] = i

    return


# -----------------------------------------------------------------------------------------
def plotStyle(L):
    """Return the plot-style (color, marker, ...) for a provided system size or
    incremental index for the matplotlib.plot()."""

    styleIndex = _style["Ldict"][L]

    return {
        "color": _style["markerEdgeColor"][styleIndex],
        "markeredgewidth": _style["markerEdgeWidth"],
        "linewidth": _style["lineWidth"],
    }


# -----------------------------------------------------------------------------------------
def errorbarStyle(L):
    """Return the plot-style (color, marker, ...) for a provided system size or
    incremental index for the matplotlib.errorbar()."""

    styleIndex = _style["Ldict"][L]

    return {
        "marker": _style["marker"][styleIndex],
        "markersize": _style["markerSize"][styleIndex],
        "color": _style["markerEdgeColor"][styleIndex],
        "mfc": _style["markerFaceColor"][styleIndex],
        "markeredgewidth": _style["markerEdgeWidth"],
        "capsize": _style["capSize"],
        "linewidth": _style["lineWidth"],
    }


# -----------------------------------------------------------------------------------------
def setTicks(ax, xTicks=[], yTicks=[], minorxTicks=None, minoryTicks=None):
    if len(xTicks) > 0:
        ax.set_xticks(xTicks)
    if len(yTicks) > 0:
        ax.set_yticks(yTicks)
    if minorxTicks != None:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(minorxTicks))
    if minoryTicks != None:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(minoryTicks))


# -----------------------------------------------------------------------------------------
def figure(figSize=(7.7, 5.6), xlabel=None, ylabel=None):

    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(111)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)

    return fig, ax


# -----------------------------------------------------------------------------------------
def legend(ax, plots, labels, legendLabelBreak=3, loc=(0.00, 0.02, 0.2), isInset=0):

    nKeyColumns = int(len(plots) / min(len(plots), legendLabelBreak) + 1)
    legendLoc = []
    if nKeyColumns > 1:
        if len(loc) == 3:
            for i in np.arange(nKeyColumns):
                legendLoc.append([loc[0] + loc[2] * i, loc[1]])
        else:
            sys.exit(
                "FS3.legend: please provide a 3-tuple for the location of the legend and the separation of the key columns."
            )
    else:
        legendLoc.append([loc[0], loc[1]])

    if isInset == 0:
        for i in range(nKeyColumns):
            plt.gca().add_artist(
                ax.legend(
                    plots[i * legendLabelBreak : (i + 1) * legendLabelBreak],
                    labels[i * legendLabelBreak : (i + 1) * legendLabelBreak],
                    loc=legendLoc[i],
                    frameon=False,
                    labelspacing=0.4,
                    handletextpad=-0.25,
                    prop={"size": mpl.rcParams["legend.fontsize"]},
                )
            )
    else:
        for i in range(nKeyColumns):
            plt.gca().add_artist(
                ax.legend(
                    plots[i * legendLabelBreak : (i + 1) * legendLabelBreak],
                    labels[i * legendLabelBreak : (i + 1) * legendLabelBreak],
                    loc=legendLoc[i],
                    frameon=False,
                    labelspacing=0.4,
                    handletextpad=-0.25,
                    prop={"size": int(mpl.rcParams["legend.fontsize"] * 0.8)},
                )
            )

    return ax


# -----------------------------------------------------------------------------------------
def addInset(fig, width=0.35, height=0.35, loc=0, xlabel="", ylabel=""):

    if loc == 0:
        ax = fig.add_axes([0.14, 0.865 - height, width, height])
        ax.xaxis.set_label_position("bottom")
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            labelleft=False,
            labelright=True,
            labeltop=False,
            labelbottom=True,
            labelsize=int(mpl.rcParams["xtick.labelsize"] * 0.8),
        )
        ax.xaxis.labelpad = -2
    elif loc == 1:
        ax = fig.add_axes([0.14, 0.14, width, height])
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            labelleft=False,
            labelright=True,
            labeltop=True,
            labelbottom=False,
            labelsize=int(mpl.rcParams["xtick.labelsize"] * 0.8),
        )
    elif loc == 2:
        ax = fig.add_axes([0.89 - width, 0.14, width, height])
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("left")
        ax.tick_params(
            labelleft=True,
            labelright=False,
            labeltop=True,
            labelbottom=False,
            labelsize=int(mpl.rcParams["xtick.labelsize"] * 0.8),
        )
    elif loc == 3:
        ax = fig.add_axes([0.89 - width, 0.865 - height, width, height])
        ax.xaxis.set_label_position("bottom")
        ax.yaxis.set_label_position("left")
        ax.tick_params(
            labelleft=True,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            labelsize=int(mpl.rcParams["xtick.labelsize"] * 0.8),
        )
        ax.xaxis.labelpad = -2
    else:
        sys.exit(
            "FS3.addInset: valid values for loc are: 0 (top left), 1 (bottom left), 2 (bottom right), 3 (top right)."
        )

    ax.set_xlabel(xlabel, fontsize=int(mpl.rcParams["axes.labelsize"] * 0.8))
    ax.set_ylabel(ylabel, fontsize=int(mpl.rcParams["axes.labelsize"] * 0.8))

    return fig, ax


# =========================================================================================
if __name__ == "__main__":
    print(
        "This is the Finite Size Scaling Suite (FS3) module, version %s" % __version__
    )
    print("Please see to the documetation at http:// for usage.\n")
