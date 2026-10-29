from dataclasses import dataclass, field, fields
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import glob
import re
import Collected_runs as cr
import FS3


def _freeze_array_fields(instance):
    """Mark every np.ndarray field of a frozen dataclass read-only.

    `frozen=True` only blocks reassigning attributes, not mutating array
    contents in place (e.g. `instance.LRange[1] = ...`), which would
    otherwise silently corrupt the shared global instance.
    """
    for f in fields(instance):
        value = getattr(instance, f.name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def sort_key(path):
    name = Path(path).name
    m = pattern.fullmatch(name)
    sign = -1 if m.group(1) == "m" else 1
    return sign * int(m.group(2))


def convert_Beta_to_2BetaNuInv(dataIn, dataNu, inverse=False):
    if len(dataIn[0]) == 3:
        alpha = dataIn[:, 0]
        if inverse:
            dataOut = 0.5 * dataNu[:, 1] * dataIn[:, 1]
        else:
            dataOut = 2 * dataIn[:, 1] / dataNu[:, 1]

        errOut = dataOut * np.sqrt(
            dataIn[:, 2] ** 2 / dataIn[:, 1] ** 2
            + dataNu[:, 2] ** 2 / dataNu[:, 1] ** 2
        )
        return np.array([alpha, dataOut, errOut]).T
    else:
        if inverse:
            dataOut = 0.5 * dataNu[:, 0] * dataIn[:, 0]
        else:
            dataOut = 2 * dataIn[:, 0] / dataNu[:, 0]

        errOut = dataOut * np.sqrt(
            dataIn[:, 1] ** 2 / dataIn[:, 0] ** 2
            + dataNu[:, 1] ** 2 / dataNu[:, 0] ** 2
        )
        return np.array([dataOut, errOut]).T


@dataclass(frozen=True)
class SelectParameters:
    """Parameters for selecting the data."""

    T0: bool
    pathRuns: np.ndarray
    alpha_list: np.ndarray
    gRange: np.ndarray
    LRange: np.ndarray
    gc_estimate: np.ndarray
    nu_estimate: np.ndarray
    z_estimate: np.ndarray
    twoBetaNuInv_estimate: np.ndarray

    def __post_init__(self):
        _freeze_array_fields(self)


@dataclass(frozen=True)
class SingleFitParameters:
    """Parameters, locations for plotting Single fits."""

    loc_legend: np.ndarray
    loc_inset: np.ndarray

    def __post_init__(self):
        _freeze_array_fields(self)


@dataclass(frozen=True)
class HistParameters:
    """Parameters for plotting Histograms."""

    x_range: np.ndarray
    x_steps: np.ndarray

    y_range: np.ndarray
    y_steps: np.ndarray

    def __post_init__(self):
        _freeze_array_fields(self)


@dataclass(frozen=True)
class HistResults:
    """Results of Histograms."""

    gc_binder: np.ndarray
    nu_binder: np.ndarray

    gc_spin_stiffness: np.ndarray
    nu_spin_stiffness: np.ndarray
    z_spin_stiffness: np.ndarray

    z_rho_U2: np.ndarray

    # z_rho_L: np.ndarray

    twoBetaNuInv_mag2_U: np.ndarray

    # twoBetaNuInv_m2_L: np.ndarray

    def __post_init__(self):
        _freeze_array_fields(self)


@dataclass(frozen=True)
class LiteratureValues:
    """Results from Literature."""

    gc: np.ndarray
    nu: np.ndarray
    beta: np.ndarray

    def __post_init__(self):
        _freeze_array_fields(self)


###########################################################################
# choose path

pathLeo4 = "/home/daniel/Master_thesis/Data/Leo4Data/"
paths = glob.glob(pathLeo4 + "*")

pattern = re.compile(r"runs_[TJ]s(m?)(\d+)")


path_runs_T = sorted(
    (p for p in paths if Path(p).name.startswith("runs_T")), key=sort_key
)

path_runs_J = sorted(
    (p for p in paths if Path(p).name.startswith("runs_J")), key=sort_key
)

alphaT = [float(2 + sort_key(path) / 100) for path in path_runs_T]
alphaJ = [float(2 + sort_key(path) / 100) for path in path_runs_J]

############################################################################
# Parameters for selecting data and initial guesses


SELECTPARAMS_T = SelectParameters(
    T0=False,
    pathRuns=np.array(path_runs_T),
    alpha_list=np.array(alphaT),
    gRange=np.array(
        [
            [2.425, 2.455],
            [1.3, 1.348],
            [1.094, 1.1],
            [0.914, 0.924],
            [0.774, 0.786],
            [0.6, 0.67],
        ]
    ),
    LRange=np.array([[64, 256], [64, 256], [64, 256], [64, 256], [64, 256], [64, 256]]),
    gc_estimate=np.array([2.436, 1.34227, 1.097, 0.917, 0.775, 0.66]),
    nu_estimate=np.array([0.976, 1.01, 1.061, 1.41, 2.0, 1.8]),
    z_estimate=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    twoBetaNuInv_estimate=np.array([1.0, 1.0, 0.8, 0.5, 0.4, 0.4]),
)

SELECTPARAMS_J = SelectParameters(
    T0=True,
    pathRuns=np.array(path_runs_J),
    alpha_list=np.array(alphaJ),
    gRange=np.array(
        [
            [9.93, 10.001],
            [5.788, 5.8005],
            [4.3225, 4.335],
            [3.5885, 3.598],
            [3.178, 3.1885],
            [2.935, 2.948],
            [2.79, 2.80],
            [2.70, 2.706],
        ],
    ),
    LRange=np.array(
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
    gc_estimate=np.array([9.987, 5.798, 4.33, 3.595, 3.183, 2.942, 2.796, 2.704]),
    nu_estimate=np.array([0.75, 0.75, 0.72, 0.71, 0.71, 0.71, 0.71, 0.71]),
    z_estimate=np.array([0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]),
    twoBetaNuInv_estimate=np.array([1.3, 1.3, 1.2, 1.12, 1.08, 1.05, 1.01, 1.01]),
)

################################################################################################
# Parameters for legend and inset locations for single fits

SINGLEFITPARAMS_BINDER_J = SingleFitParameters(
    loc_inset=np.array([3, 3, 3, 3, 3, 3, 3, 3]),
    loc_legend=np.array(
        [
            (0.25, 0.6 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_BINDER_T = SingleFitParameters(
    loc_inset=np.array([1, 1, 1, 1, 1, 1]),
    loc_legend=np.array(
        [
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_SPIN_STIFFNESS_J = SingleFitParameters(
    loc_inset=np.array([3, 3, 3, 3, 3, 3, 3, 3]),
    loc_legend=np.array(
        [
            (0.25, 0.6 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
            (0.02, 0.02 + 0.4, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_SPIN_STIFFNESS_T = SingleFitParameters(
    loc_inset=np.array([3, 3, 3, 3, 3, 1]),
    loc_legend=np.array(
        [
            (0.02, 0.02 + 0.25, 0.18),
            (0.02, 0.02 + 0.25, 0.18),
            (0.02, 0.02 + 0.25, 0.18),
            (0.02, 0.02 + 0.25, 0.18),
            (0.02, 0.02 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_RHO_U2_J = SingleFitParameters(
    loc_inset=np.array([0, 2, 3, 3, 0, 0, 0, 3]),
    loc_legend=np.array(
        [
            (0.75, 0.02 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.75, 0.6 + 0.4, 0.18),
            (0.75, 0.6 + 0.4, 0.18),
            (0.75, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_RHO_U2_T = SingleFitParameters(
    loc_inset=np.array([0, 0, 0, 0, 0, 0]),
    loc_legend=np.array(
        [
            (0.75, 0.02 + 0.25, 0.18),
            (0.75, 0.02 + 0.25, 0.18),
            (0.75, 0.02 + 0.25, 0.18),
            (0.75, 0.02 + 0.25, 0.18),
            (0.75, 0.02 + 0.25, 0.18),
            (0.75, 0.02 + 0.25, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_BETA_J = SingleFitParameters(
    loc_inset=np.array([3, 3, 3, 3, 3, 0, 0, 3]),
    loc_legend=np.array(
        [
            (0.25, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
            (0.75, 0.6 + 0.4, 0.18),
            (0.75, 0.6 + 0.4, 0.18),
            (0.02, 0.6 + 0.4, 0.18),
        ]
    ),
)

SINGLEFITPARAMS_BETA_T = SingleFitParameters(
    loc_inset=np.array([3, 0, 0, 0, 0, 0]),
    loc_legend=np.array(
        [
            (0.02, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
            (0.77, 0.75 + 0.25, 0.18),
            (0.75, 0.75 + 0.25, 0.18),
        ]
    ),
)

################################################################################################
# Histogram Parameters plotting

HISTPARAMS_BINDER_J = HistParameters(
    x_range=np.array(
        [
            [0.729, 0.771],
            [0.705, 0.745],
            [0.705, 0.735],
            [0.695, 0.725],
            [0.695, 0.725],
            [0.69, 0.725],
            [0.69, 0.74],
            [0.68, 0.74],
        ]
    ),
    x_steps=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]),
    y_range=np.array(
        [
            [9.983, 9.991],
            [5.7949, 5.7981],
            [4.3299, 4.3316],
            [3.5944, 3.5956],
            [3.18329, 3.18371],
            [2.9421, 2.9429],
            [2.7954, 2.7961],
            [2.7035, 2.7045],
        ]
    ),
    y_steps=np.array([0.002, 0.001, 0.0005, 0.0005, 0.0001, 0.0002, 0.0002, 0.0002]),
)

HISTPARAMS_BINDER_T = HistParameters(
    x_range=np.array(
        [
            [0.96, 1.02],
            [0.98, 1.01],
            [1.025, 1.125],
            [1.2, 1.4],
            [1.5, 4.0],
            [2.0, 25.0],
        ]
    ),
    x_steps=np.array([0.02, 0.01, 0.05, 0.1, 1.0, 5.0]),
    y_range=np.array(
        [
            [2.4325, 2.4375],
            [1.342, 1.343],
            [1.096, 1.0975],
            [0.917, 0.919],
            [0.766, 0.778],
            [0.6, 0.7],
        ]
    ),
    y_steps=np.array([0.001, 0.0005, 0.0005, 0.0005, 0.005, 0.05]),
)

HISTPARAMS_SPIN_STIFFNESS_NU_J = HistParameters(
    x_range=np.array(
        [
            [0.76, 0.8],
            [0.75, 0.8],
            [0.705, 0.75],
            [0.70, 0.73],
            [0.71, 0.73],
            [0.70, 0.73],
            [0.7, 0.74],
            [0.7, 0.75],
        ]
    ),
    x_steps=np.array([0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02]),
    y_range=np.array(
        [
            [9.979, 9.981],
            [5.7947, 5.7952],
            [4.3294, 4.3298],
            [3.5938, 3.5942],
            [3.18275, 3.183],
            [2.94205, 2.94225],
            [2.7955, 2.796],
            [2.7036, 2.7042],
        ]
    ),
    y_steps=np.array([0.0005, 0.0002, 0.0002, 0.0001, 0.0001, 0.00005, 0.0002, 0.0002]),
)

HISTPARAMS_SPIN_STIFFNESS_NU_T = HistParameters(
    x_range=np.array(
        [
            [0.95, 0.97],
            [1.0, 1.02],
            [1.03, 1.06],
            [1.22, 1.24],
            [1.8, 1.95],
            [0.0, 50.0],
        ]
    ),
    x_steps=np.array([0.01, 0.01, 0.01, 0.01, 0.05, 5.0]),
    y_range=np.array(
        [
            [2.4352, 2.4355],
            [1.34205, 1.34225],
            [1.09645, 1.09665],
            [0.9174, 0.9176],
            [0.7725, 0.7735],
            [0.0, 0.7],
        ]
    ),
    y_steps=np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0005, 0.1]),
)

HISTPARAMS_SPIN_STIFFNESS_Z_J = HistParameters(
    x_range=np.array(
        [
            [-0.275, -0.26],
            [0.222, 0.238],
            [0.605, 0.618],
            [0.819, 0.8351],
            [0.9275, 0.9425],
            [0.9649, 0.981],
            [0.9675, 0.9925],
            [0.97, 1.0],
        ]
    ),
    x_steps=np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01]),
    y_range=np.array(
        [
            [-0.3, 0.5],
            [-0.4, 0.4],
            [-0.35, 0.35],
            [-0.4, 0.4],
            [-0.4, 0.4],
            [-0.4, 0.4],
            [-0.4, 0.44],
            [-0.4, 0.4],
        ]
    ),
    y_steps=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
)

HISTPARAMS_SPIN_STIFFNESS_Z_T = HistParameters(
    x_range=np.array(
        [
            [-0.6025, -0.5925],
            [-0.145, -0.135],
            [-0.033, -0.024],
            [-0.005, 0.005],
            [-0.02, 0.0],
            [-0.15, -0.05],
        ]
    ),
    x_steps=np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.05]),
    y_range=np.array(
        [[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3], [-0.4, 0.4], [-0.3, 0.3], [-0.4, 0.4]]
    ),
    y_steps=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
)

HISTPARAMS_RHO_U2_Z_J = HistParameters(
    x_range=np.array(
        [
            [-0.2, 0.02],
            [0.38, 0.42],
            [0.74, 0.78],
            [0.9, 1.02],
            [0.97, 1.15],
            [1.04, 1.08],
            [0.94, 1.07],
            [1.04, 1.12],
        ]
    ),
    x_steps=np.array([0.1, 0.02, 0.02, 0.05, 0.05, 0.02, 0.05, 0.02]),
    y_range=np.array(
        [
            [0, 1.2],
            [0.14, 0.22],
            [0.3, 0.5],
            [0.4, 1.1],
            [0, 1.1],
            [0.25, 0.5],
            [0.25, 4],
            [0, 0.5],
        ]
    ),
    y_steps=np.array([0.5, 0.02, 0.1, 0.2, 0.5, 0.1, 1.0, 0.1]),
)

HISTPARAMS_RHO_U2_Z_T = HistParameters(
    x_range=np.array(
        [
            [-0.44, -0.35],
            [-0.05, -0.015],
            [-0.05, 0.05],
            [-0.3, 0.1],
            [-0.1, 0.1],
            [-0.2, 0.1],
        ]
    ),
    x_steps=np.array([0.02, 0.01, 0.02, 0.1, 0.05, 0.1]),
    y_range=np.array(
        [[0.15, 0.25], [0.125, 0.25], [-0.5, 6.0], [0.0, 0.9], [0, 0.8], [0, 2.0]]
    ),
    y_steps=np.array([0.05, 0.05, 1.0, 0.2, 0.2, 0.2]),
)


HISTPARAMS_BETA_J = HistParameters(
    x_range=np.array(
        [
            [1.32, 1.39],
            [1.3, 1.35],
            [1.21, 1.23],
            [1.11, 1.17],
            [1.05, 1.085],
            [1.025, 1.06],
            [1.007, 1.015],
            [1.0, 1.08],
        ]
    ),
    x_steps=np.array([0.02, 0.02, 0.01, 0.02, 0.01, 0.01, 0.002, 0.02]),
    y_range=np.array(
        [
            [0.0, 3.5],
            [0.5, 1.3],
            [1.0, 1.5],
            [0.2, 1.2],
            [0, 4.0],
            [0, 4.0],
            [5, 11.0],
            [0, 4],
        ]
    ),
    y_steps=np.array([1.0, 0.2, 0.5, 0.2, 1, 1, 2, 1]),
)

HISTPARAMS_BETA_T = HistParameters(
    x_range=np.array(
        [
            [0.98, 1.02],
            [0.89, 0.92],
            [0.76, 0.77],
            [0.53, 0.57],
            [0.38, 0.45],
            [-0.2, 0.4],
        ]
    ),
    x_steps=np.array([0.02, 0.01, 0.005, 0.01, 0.02, 0.2]),
    y_range=np.array(
        [
            [0.0, 2],
            [0.5, 2.0],
            [4.0, 7.0],
            [0.0, 2.0],
            [0, 3.0],
            [0, 5.0],
        ]
    ),
    y_steps=np.array([1, 0.5, 1, 1, 1, 1]),
)


################################################################################################
# Histogram Results

HISTRESULTS_J = HistResults(
    gc_binder=np.array(
        [
            [9.986811, 0.000486],
            [5.796548, 0.000299],
            [4.330416, 0.000127],
            [3.595128, 0.000051],
            [3.183534, 0.000041],
            [2.942463, 0.000140],
            [2.795655, 0.000047],
            [2.703704, 0.000082],
        ]
    ),
    nu_binder=np.array(
        [
            [0.752165, 0.003444],
            [0.727409, 0.007701],
            [0.717610, 0.003892],
            [0.708379, 0.004396],
            [0.710241, 0.003442],
            [0.703904, 0.004905],
            [0.714799, 0.005880],
            [0.712447, 0.010200],
        ]
    ),
    gc_spin_stiffness=np.array(
        [
            [9.980339, 0.000174],
            [5.794953, 0.000058],
            [4.329596, 0.000037],
            [3.594004, 0.000033],
            [3.182863, 0.000030],
            [2.942135, 0.000028],
            [2.795722, 0.000058],
            [2.703874, 0.000051],
        ]
    ),
    nu_spin_stiffness=np.array(
        [
            [0.782428, 0.002731],
            [0.773378, 0.004708],
            [0.728809, 0.003447],
            [0.713844,0.003627],
            [0.720367, 0.002668],
            [0.717652, 0.002074],
            [0.718202, 0.004409],
            [0.723739, 0.006952],
        ]
    ),
    z_spin_stiffness=np.array(
        [
            [-0.267237, 0.000895],
            [0.229030, 0.001221],
            [0.611782, 0.001613],
            [0.827408, 0.002000],
            [0.934670, 0.001983],
            [0.972330, 0.001935],
            [0.979337, 0.003234],
            [0.983914, 0.002918],
        ]
    ),
    z_rho_U2=np.array(
        [
            [-0.077003, 0.038253],
            [0.400293, 0.004286],
            [0.765522, 0.004634],
            [0.953320, 0.006430],
            [1.017063, 0.007105],
            [1.057453, 0.005060],
            [1.010736, 0.022482],
            [1.085485, 0.010752],
        ]
    ),
    twoBetaNuInv_mag2_U=np.array(
        [
            [1.363548, 0.014931],
            [1.307202, 0.001776],
            [1.219666, 0.001661],
            [1.130849, 0.003915],
            [1.070354, 0.003984],
            [1.048019, 0.004055],
            [1.011261, 0.000942],
            [1.024475, 0.002969],
        ]
    ),
)

HISTRESULTS_T = HistResults(
    nu_binder=np.array(
        [
            [0.997480, 0.008193],
            [0.996875, 0.005244],
            [1.066885, 0.008656],
            [1.310429, 0.027550],
            [1.910010, 0.102308],
            [8.661993, 2.384114],
        ]
    ),
    gc_binder=np.array(
        [
            [2.436320, 0.000177],
            [1.342539, 0.000073],
            [1.096670, 0.000056],
            [0.917964, 0.000194],
            [0.776129, 0.000615],
            [0.650344, 0.004519],
        ]
    ),
    nu_spin_stiffness=np.array(
        [
            [0.958626, 0.002524],
            [1.009266, 0.002023],
            [1.044320, 0.002935],
            [1.230769, 0.002393],
            [1.863325, 0.010418],
            [14.787814, 4.651267],
        ]
    ),
    gc_spin_stiffness=np.array(
        [
            [2.435316, 0.000034],
            [1.342142, 0.000020],
            [1.096545, 0.000020],
            [0.917518, 0.000023],
            [0.773050, 0.000088],
            [0.468090, 0.068431],
        ]
    ),
    z_spin_stiffness=np.array(
        [
            [-0.596622, 0.001168],
            [-0.139009, 0.001429],
            [-0.028231, 0.001478],
            [-0.000192, 0.001068],
            [-0.010823, 0.001176],
            [-0.096016, 0.008759],
        ]
    ),
    z_rho_U2=np.array(
        [
            [-0.397709, 0.011740],
            [-0.031547, 0.005415],
            [0.010110, 0.005278],
            [0.007072, 0.017559],
            [0.000100, 0.036235],
            [-0.0156,0.0001],
        ]
    ),
    twoBetaNuInv_mag2_U=np.array(
        [
            [1.000636, 0.004498],
            [0.900402, 0.003245],
            [0.766723, 0.000618],
            [0.550922, 0.002859],
            [0.421735, 0.003752],
            [0.211497,0.029402],
        ]
    ),
)

nuJ_Lang_alpha2 = [2.0, 0.737426, 0.007683]

################################################################################################
# Literature values

LITVALUES_T = LiteratureValues(
    gc=np.array(
        [  # alpha - 1, mean, err
            [2.30, 5.68075223, 0.06713568],
            [2.50, 3.57807943, 0.03346261],
            [2.70, 2.60208653, 0.02104567],
            [2.90, 2.02985478, 0.02104567],
            [3.00, 1.82160789, 0.02083521],
            [3.05, 1.72427167, 0.02104567],
            [3.10, 1.64471904, 0.02104567],
            [3.15, 1.55590632, 0.02104567],
            [3.20, 1.48677129, 0.00947055],
            [3.30, 1.34808033, 0.00947055],
            [3.40, 1.23264484, 0.00926009],
            [3.50, 1.12815309, 0.00947055],
            [3.60, 1.01587445, 0.00926009],
            [3.70, 0.93021857, 0.00926009],
        ]
    ),
    nu=np.array(
        [  # alpha, mean, err
            [2.30, 1.02932108, 0.03546215],
            [2.50, 1.02949726, 0.03533631],
            [2.70, 1.00060404, 0.03493362],
            [2.90, 0.99131693, 0.03495879],
            [3.00, 1.00113257, 0.03168691],
            [3.05, 1.01031901, 0.03488328],
            [3.10, 1.02081419, 0.03128421],
            [3.15, 1.03100736, 0.03115837],
            [3.20, 1.03445542, 0.03113320],
            [3.30, 1.09020323, 0.03855786],
            [3.40, 1.11079091, 0.07364248],
            [3.50, 1.27936828, 0.07973322],
            [3.60, 1.49898698, 0.11839174],
            [3.70, 1.63819292, 0.12581640],
        ]
    ),
    beta=np.array(
        [  # alpha, mean, err
            [2.30, 0.52924515, 0.03442599],
            [2.50, 0.53276764, 0.03373602],
            [2.70, 0.52270857, 0.03476493],
            [2.90, 0.48908149, 0.03500702],
            [3.00, 0.50858229, 0.03129085],
            [3.05, 0.49458916, 0.03475282],
            [3.10, 0.44402750, 0.03113349],
            [3.15, 0.43905244, 0.03141190],
            [3.20, 0.39971191, 0.03112139],
            [3.30, 0.39729095, 0.03809374],
            [3.40, 0.31500266, 0.04201569],
            [3.50, 0.31367114, 0.04538082],
            [3.60, 0.31471215, 0.05574251],
            [3.70, 0.31986878, 0.06959037],
        ]
    ),
)

LITVALUES_J = LiteratureValues(
    gc=np.array(
        [
            [2.503793, 14.733212, 0],
            [2.804248, 9.949183, 0],
            [3.001517, 8.257713, 0],
            [3.101669, 7.633394, 0],
            [3.204855, 7.052632, 0],
            [3.405159, 6.275862, 0],
            [3.502276, 5.941924, 0],
            [3.805766, 5.143376, 0],
            [4.003035, 4.751361, 0],
            [4.503794, 4.076225, 0],
            [5.007587, 3.655173, 0],
            [6.006069, 3.168784, 0],
            [8.003035, 2.805808, 0],
            [10.006069, 2.660617, 0],
        ]
    ),
    nu=np.array(
        [
            [2.50, 0.750483, 0.004128],
            [2.80, 0.750672, 0.004160],
            [3.00, 0.747048, 0.005515],
            [3.10, 0.750515, 0.005074],
            [3.20, 0.751113, 0.004601],
            [3.40, 0.742889, 0.002773],
            [3.50, 0.734632, 0.003908],
            [3.80, 0.714559, 0.002426],
            [4.00, 0.707626, 0.002962],
            [4.50, 0.708004, 0.001985],
            [5.00, 0.709989, 0.002994],
            [6.00, 0.709989, 0.002994],
        ]
    ),
    beta=np.array(
        [
            [2.50, 0.497140, 0.016121],
            [2.80, 0.499220, 0.013781],
            [3.00, 0.496880, 0.017161],
            [3.10, 0.490769, 0.011701],
            [3.20, 0.500910, 0.013391],
            [3.40, 0.486349, 0.011701],
            [3.50, 0.441235, 0.011181],
            [3.80, 0.392221, 0.011831],
            [4.00, 0.385720, 0.009881],
            [4.50, 0.370899, 0.007151],
            [5.00, 0.367909, 0.006240],
            [6.00, 0.365309, 0.005720],
        ]
    ),
)
