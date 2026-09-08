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

# from numpy.polynomial.chebyshev import chebval
# import parameters
# from cycler import cycler

# mpl.rcParams.update(
#     {
#         "text.usetex": False,
#         "font.family": "serif",
#         "font.serif": "cmr10",
#         "mathtext.fontset": "cm",
#         "font.family": "STIXGeneral",
#         "axes.unicode_minus": True,
#         "axes.labelsize": 21,
#         "xtick.labelsize": 18,
#         "ytick.labelsize": 18,
#         "legend.fontsize": 16,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "xtick.major.top": "on",
#         "xtick.major.bottom": "on",
#         "ytick.major.left": "on",
#         "ytick.major.right": "on",
#         "xtick.top": True,
#         "ytick.right": True,
#         "figure.autolayout": True,
#     }
# )

nupZhao23 = np.array(
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
)

betaZhao23 = np.array(
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
)

etaZhao23 = np.array(
    [  # alpha, mean, err
        [2.30, 1.00482525, 0.02614361],
        [2.50, 1.00591571, 0.02854261],
        [2.70, 1.00635189, 0.01927376],
        [2.90, 0.99920942, 0.02554386],
        [3.00, 1.01951911, 0.02306308],
        [3.05, 0.96927648, 0.02189085],
        [3.10, 0.92233248, 0.02276321],
        [3.15, 0.88670192, 0.01019574],
        [3.20, 0.79055122, 0.02090944],
        [3.30, 0.71830871, 0.02041873],
        [3.40, 0.61425222, 0.02088218],
        [3.50, 0.49795540, 0.01984625],
        [3.60, 0.41900660, 0.02000981],
        [3.70, 0.38471185, 0.02219072],
    ]
)

L_list = [32, 64, 128, 256, 512,1024, 2048, 4096]
JSLAC_list = []
JFC_list = []

alpha = 2.0

for L in L_list:
    JSLAC = 0
    JFC = 0
    for x in range(1, L // 2):
        JSLAC += 4 / (x**alpha)
        JFC+= 4 / (x**alpha)
        for y in range(1, L // 2):
            JFC += 4 / ((x**2 + y**2) ** (0.5 * (alpha+1)))
    JSLAC_list.append(JSLAC)
    JFC_list.append(JFC)


plt.scatter(L_list, np.array(JSLAC_list))
plt.scatter(L_list, np.array(JFC_list))

plt.show()
