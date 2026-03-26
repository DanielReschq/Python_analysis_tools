import numpy as np
from numpy.random import random as ran
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

L = 8

spin = np.random.randint(0, 2, size=L) * 2 - 1

bonds = np.ndarray(shape=(L, 2), dtype=int)
for i in range(L):
    bonds[i] = np.array([i, (i + 1) % L])

def state(spin):

    state = spin * range(1, L + 1)
    state_up = state[state > 0]
    state_up = state_up - np.ones(len(state_up))
    state_down = -state[state < 0]
    state_down = state_down - np.ones(len(state_down))
    return state_up, state_down

m = 10
beta = 6*L

op_string = np.zeros(m, dtype=int)
n_op = 0


def diag_update():
    global op_string, n_op, spin
    op_string_poss = np.zeros(len(op_string), dtype=int)

    x = 0.5 * L * beta

    for i in range(m):
        op = op_string[i]
        if op == 0:  # identity operator present
            b = int(ran() * L)
            op_string_poss[i] = 2 * (b + 1)
            if spin[bonds[b, 0]] != spin[bonds[b, 1]]:
                if ran() < x/(m-n_op):  # add operator with P([0,0] -> [1,b]), n -> n+1
                    op_string[i] = 2 * (b + 1)
                    n_op += 1
        elif op % 2 == 0:  # diagonal operator present
            op_string_poss[i] = op_string[i]
            if ran() < (m-n_op + 1)/x:  # remove operator with P([1,b] -> [0,0]), n -> n-1
                op_string[i] = 0
                n_op -= 1
        else:  # off-diagonal operator present, just propagate the state
            b = int(op // 2 - 1)
            spin[bonds[b]] *= -1
            spin[bonds[b, 1]] *= -1
    return op_string_poss


def vertex_list():
    global v_first
    global v_last
    v_list = -np.ones(4 * m, dtype=int)


    for p in range(0, m):
        op = op_string[p]
        v0 = 4 * p

        if op != 0:
            bond_index = int(op // 2 - 1)
            i = bonds[bond_index, 0]
            j = bonds[bond_index, 1]

            v1 = v_last[i]
            v2 = v_last[j]

            if v1 == -1: #site never visited
                v_first[i] = v0
            else:
                v_list[v0] = v1
                v_list[v1] = v0

            if v2 == -1: #site never visited
                v_first[j] = v0 + 1
            else:
                v_list[v0 + 1] = v2
                v_list[v2] = v0 + 1

            v_last[i] = v0 + 2
            v_last[j] = v0 + 3

        else:
            v_list[v0 : v0 + 3] = -1 #no site visited this p

        for i in range(L):
            v1 = v_first[i]
            if v1 != -1:
                v2 = v_last[i]
                v_list[v1] = v2
                v_list[v2] = v1
    return v_list

def loop_update():
    global op_string
    global v_list

    global v_first
    global v_last

    for v0 in range(0,4*m,2):

        #print(f"start: {v0}")
        if v_list[v0] < 0: # already visited or no operator
            #print(f"already visited or no operator")
            continue

        ingoing = v0

        flip_status = -1

        if (np.random.uniform(0,1.) < 0.5):
            flip_status = -2

        while(True):
                if (flip_status == -2):
                    #print("flip")
                    p = ingoing//4
                    op_string[p] = op_string[p]^1 #switch operator
                outgoing = ingoing^1
                #print(f"-> {outgoing}")
                ingoing = v_list[outgoing]
                v_list[outgoing] = flip_status
                v_list[ingoing] = flip_status
                #print(f"-> {ingoing}")

                if(ingoing == v0):
                    #print("loop terminated")
                    break

    for i in range(L):
        if(v_list[v_first[i]] == -2):
            spin[i] *= -1
        elif(v_first[i] == -1):
            if (np.random.uniform(0,1.) < 0.5):
                spin[i] *= -1

def op_color(op,poss = False):
    if not poss:
        if (op%2 == 0):
            return "red","-"
        else:
            return "green","-"
    if (op%2 == 0):
        return "blue","--"
    else:
        return "yellow","--"

def draw_opstring(op_string,ax,poss):
    global spin
    for i in range(L):
        ax.axvline(x=i, ymin=1 / m, ymax=(m - 1) / m, color="black", ls="--", alpha=0.5)

    state_up,state_down = state(spin)
    ax.scatter(state_up, 0 * np.ones(len(state_up)), marker="x", color="black")
    ax.scatter(state_down, 0 * np.ones(len(state_down)), marker="o", color="black")

    for p in range(m):
        op = op_string[p]
        if (op == 0):
            continue
        if (op != 0):
            bond_index = int(op // 2 - 1)
            i = bonds[bond_index, 0]
            j = bonds[bond_index, 1]
            ax.scatter(state_up, p * np.ones(len(state_up)), marker="x", color="black")
            ax.scatter(state_down, p * np.ones(len(state_down)), marker="o", color="black")

            xmin = (i + 1) / (L + 1)
            xmax=(j + 1) / (L + 1)

            if (i < j):
                ax.axhline(
                    y=p + 0.5, xmin=xmin, xmax=xmax, color=op_color(op,poss)[0], ls = op_color(op,poss)[1]
                )
            else:
                ax.axhline(
                    y=p + 0.5, xmin=xmin, xmax=xmin + 0.5/(L+2), color=op_color(op,poss)[0], ls = op_color(op,poss)[1]
                )
                ax.axhline(
                    y=p + 0.5, xmin=xmax - 0.5/(L+2), xmax=xmax, color=op_color(op,poss)[0], ls = op_color(op,poss)[1]
                )

            if (op %2 != 0):
                spin[i] *= -1
                spin[j] *= -1
                state_up,state_down = state(spin)

            ax.scatter(
                state_up, (p + 1) * np.ones(len(state_up)), marker="x", color="black"
            )
            ax.scatter(
                state_down, (p + 1) * np.ones(len(state_down)), marker="o", color="black"
            )

    ax.scatter(state_up, (m) * np.ones(len(state_up)), marker="x", color="black")
    ax.scatter(state_down, (m) * np.ones(len(state_down)), marker="o", color="black")

fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize = (10,4))
ax1.set(xlim = (-1, L), ylim = (-1, m+1))
ax2.set(xlim = (-1, L), ylim = (-1, m+1))
ax3.set(xlim = (-1, L), ylim = (-1, m+1))


diag_update()
v_first = -np.ones(L, dtype=int)
v_last = -np.ones(L, dtype=int)
v_list = vertex_list()
loop_update()

draw_opstring(op_string,ax1,False)
op_poss = diag_update()
draw_opstring(op_poss,ax1,True)
draw_opstring(op_string,ax2,False)
ax1.set_title("before diagonal update")
ax2.set_title("after diagonal update")

v_first = -np.ones(L, dtype=int)
v_last = -np.ones(L, dtype=int)
v_list = vertex_list()
loop_update()
loop_update()
draw_opstring(op_string,ax3,False)
ax3.set_title("after loop update")

legend_handles = [Patch(color="blue", label='chosen diag. op'),  # red
                  Patch(color="red", label='diag. op.'),
                  Patch(color="green", label='offdiag. op.')]  # green
plt.legend(handles=legend_handles, ncol=1, bbox_to_anchor=[1.3, 0.9], loc='upper center', fontsize=8, handlelength=.8)

fig.savefig("diag_loop_update.pdf")
plt.show()