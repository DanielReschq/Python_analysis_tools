import numpy as np
import matplotlib.pyplot as plt
import itertools

L = 16

def delta(point):
    delta_x,delta_y = np.meshgrid(np.zeros(L),np.zeros(L))

    for i in range(0,L):
        delta_x[i,point[0]] = 1
        delta_y[point[1],i] = 1

    return delta_x,delta_y

def plot_connections(point):

    x = np.linspace(0,L-1,L)
    y = np.linspace(0,L-1,L)

    X,Y = np.meshgrid(x,y)

    delta_x,delta_y = delta(point)
        
    R = np.ndarray(shape=(L,L),dtype=int)

    R = np.sqrt((X-point[0])**2 + (Y-point[1])**2)

    t_ij = 1/((L/np.pi)**2*np.sin(np.pi*R/L)**2)*(delta_x+delta_y)
    t_ij[point[0],point[1]] = 0
    
    plt.imshow(t_ij,cmap="bwr")
    plt.colorbar()
    plt.clim(-1,1)
    plt.show()

def plot_compare_inf(point):
    x = np.linspace(0,L-1,L)
    r = np.zeros(L)
    for i in range(L):
        rad = abs(x[i]-point[0])
        r[i] = np.min([rad,abs(L-rad)])
        
    t = np.pi/L*1/np.sin(np.pi*r/L)
    J = t**2

    plt.scatter(x,J)
    plt.plot(x,(1/r)**2)
    plt.show()

point = [0,1]

plot_connections(point)
plot_compare_inf(point)

def plot_disp_rel():
    ax = plt.figure().add_subplot(projection="3d")
    k_x = np.linspace(-np.pi, np.pi, L)
    k_y = np.linspace(-np.pi, np.pi, L)

    kX, kY = np.meshgrid(k_x, k_y)

    eta_x = np.linspace(1, L - 1, L - 1)
    eta_y = np.linspace(1, L - 1, L - 1)

    x = np.array(list(itertools.product(eta_x, [0])))
    y = np.array(list(itertools.product([0], eta_y)))

    eta = np.concatenate(
        (x,y)
    )

    E_k = 0*kX
    for i in range(0,len(eta)):
        eta_norm = max(eta[i])
        t = 1/((L/np.pi)*np.sin(np.pi*(eta_norm)/L))
        E_k +=t**2*np.cos(kX*eta[i,0]+kY*eta[i,1])

    ax.plot_surface(kX, kY, E_k)
    plt.show()

#plot_disp_rel()
