import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

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


def t(eta):
    t = (np.pi) / L * (-1.0) ** (eta) * 1 / (np.sin(np.pi / L * eta))
    return t


L = 1001
k_x = np.linspace(-np.pi, np.pi, L)
k_y = np.linspace(-np.pi, np.pi, L)
kX, kY = np.meshgrid(k_x, k_y)

E_kp = np.sqrt(kX**2 + kY**2)
E_km = -np.sqrt(kX**2 + kY**2)

E_kpU = np.sqrt((kX**2 + kY**2 + 0.5))
E_kmU = -np.sqrt((kX**2 + kY**2 + 0.5))

ax = plt.figure().add_subplot(projection="3d")
ax.plot_surface(kX, kY, E_kpU)
ax.plot_surface(kX, kY, E_kmU)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_zlabel(r"$E(\mathbf{k})$")
# plt.savefig("Dispersion.pdf")
plt.show()
plt.close()

x = np.array(range(1, L // 2))
plt.plot(x, t(x), label=r"$t(R)$", color="red")
plt.plot(x, (np.pi * 1.0) / np.pi / x, label=r"$\propto 1/R$", color="black", ls="--")
plt.plot(x, (np.pi * 1.0) / (-1 * np.pi) / x, color="black", ls="--")
plt.legend()
plt.xlabel("R")
plt.ylabel("t")
# plt.savefig("t_R.pdf")
plt.show()
plt.close()

Rx, Ry = np.meshgrid(np.linspace(0, L - 1, L), np.linspace(0, L - 1, L))
H_real = np.zeros(shape=(L, L))
H_imag = np.zeros(shape=(L, L))
lamb = np.zeros(shape=(L, L))
eps = 0
for R in range(1, L // 2 + 1):
    H_real[L // 2, L // 2 + R] = -t(R)
    H_real[L // 2, L // 2 - R] = t(R)
    H_imag[L // 2 + R, L // 2] = t(R)
    H_imag[L // 2 - R, L // 2] = -t(R)
    lamb += 2 * (np.cos(kX * R) / R**2 + np.cos(kY * R) / R**2)
    eps += 4 / R**2
plt.imshow(H_real, cmap="bwr")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar().set_label("t")
#plt.savefig("MA_real.pdf")
plt.imshow(H_imag, cmap="bwr")
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig("MA_imag.pdf")
plt.close()

EkS = np.sqrt(eps**2 - lamb**2)
ax = plt.figure().add_subplot(projection="3d")
ax.plot_surface(kX, kY, EkS)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_zlabel(r"$E(\mathbf{k})$")
# plt.savefig("Disp_boson.pdf")
plt.show()
plt.close()

Rand = np.random.randint(0,2,size=(L,L))
AF = (-1) ** (Rx + Ry)
fig,ax = plt.subplots()
ax.imshow(Rand,cmap="gist_gray")

legend_handles = [Patch(color="black", label='Spin up'),  # red
                  Patch(color="white", label='Spin down')]  # green
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)
plt.tight_layout()
#plt.savefig("Rand.pdf")
plt.show()
plt.close()

x = np.linspace(0,1.2,1000)
y = np.zeros_like(x)
y[x<1.0] = 3.142*np.sqrt(1-(x[x<1.0])**2)
fig, ax = plt.subplots()
ax.plot(x,y)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[5] = '?'

ax.set_xticklabels(labels)
ax.errorbar(0,3.14196,yerr=0.00001,label="Data",marker = "s", color = "black")
ax.fill_between(x, y, 0, color='blue', alpha=.1)
ax.fill_between(x, y, 4, color='red', alpha=.1)

ax.text(0.3,1.0,"AF",fontsize=20)
ax.text(1.0,3.0,"PM",fontsize=20)
ax.set_xlim(0,1.2)
ax.set_ylim(0,4)
ax.set_xlabel(r"$J_{\perp}$")
ax.set_ylabel(r"$T$")
plt.legend()
plt.show()
plt.close()
