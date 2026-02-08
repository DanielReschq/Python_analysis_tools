#
#  Example:
#
#  a1, a2 = np.array([1., 0.]), np.array([0., 1.])
#  a1, a2 = np.array([ np.sqrt(3), 0]), np.array([ np.sqrt(3)/2, 3/2])
#  
#  simTorus = np.array([[1, 4], [5, -2]])
#  latt = lattice2D(a1, a2, simTorus)
#  latt.plot()
#

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

__version__ = '1.0.0'


#=========================================================================================
class create:
    #-----------------------------------------------------------------------------------------
    def __init__(self, a1, a2, simTorus, a=1.0, PBC=(True,True)):

        """Creates the Wigner Seitz cell vertices, real space lattice sites within, the 
        vertices of the 1st Brioullin zone and momenta within.
        
        self.a                     lattice constant
        self.simTorus[n, m]        simulation torus, defined in integer multiples of a1, a2
        self.A1, self.A2           primitive vectors of the simulation torus
        self.STcell                vertices of the simulation torus
        self.WScell                vertices of the Wigner Seitz cell (if applicable)
        self.b1, self.b2           primitive vectors of the reciprocal lattice
        self.B1, self.B2           primitive vectors of the Brioullin zone
        self.BZone                 vertices of the Brioullin zone
        self.PBC[True,True]        boundary conditions on a1, a2 direction
        self.N                     number of unit cells, or momenta
        self.list[uc]              real space locations, defined in integer multiples of a1, a2
        self.invlist[n, m]         unit cell index, provided the integer multiples of a1, a2
        self.invlistmin            index offset for the above
        self.klist[uc]             momenta, defined in integer multiples of b1, b2
        self.invklist[n, m]        momentum index, provided the integer multiples of b1, b2
        self.invklistmin           index offset for the above
        self.nn[uc, +/-a1, +/-a2]  nearest neighbor unit cell indices in +/-a1, +/-a2 direction
        """

        self.a = a
        self.a1 = a1.astype(dtype=np.double) * self.a
        self.a2 = a2.astype(dtype=np.double) * self.a
        self.simTorus = simTorus.astype(dtype=np.int_)
        self.PBC = PBC
        
    #   Wigner Seitz cell
        self.A1 = self.simTorus[0,0]*self.a1 + self.simTorus[0,1]*self.a2
        self.A2 = self.simTorus[1,0]*self.a1 + self.simTorus[1,1]*self.a2
        self.STcell = np.array([[0,0], self.A1, self.A1+self.A2, self.A2], dtype=np.double)
        self.WScell = tesselate(self.A1, self.A2)
        d = int(max(np.linalg.norm(self.A1), np.linalg.norm(self.A2))/min(np.linalg.norm(self.a1), np.linalg.norm(self.a2)))
        self.list = pointsWithin(d, self.WScell, self.a1, self.a2, (1E-8*(self.a1/5+self.a2/3)))
        self.list_xy = np.dot(self.list, [self.a1, self.a2])
        self.N = len(self.list)        

        nmin, nmax = np.amin(self.list[:,0]), np.amax(self.list[:,0])
        mmin, mmax = np.amin(self.list[:,1]), np.amax(self.list[:,1])
        self.invlistmin = np.array([nmin, mmin])
        self.invlist = -np.ones((np.abs(nmax-nmin)+1, np.abs(mmax-mmin)+1), dtype=int)
        i = 0
        for i in range(len(self.list)):
            nm = self.list[i] - self.invlistmin
            self.invlist[nm[0], nm[1]] = i
            i += 1

    #   nearest neighbors in directions +/0/-a1, +/0/-a2
        self.nn = -np.ones((self.N, 3, 3), dtype=int)
        for iP in range(self.N):
            nmP = np.array([self.list[iP,0] - self.invlistmin[0], self.list[iP,1] - self.invlistmin[1]])
            nmP = self.list[iP]
            dmin = 99999999
            iQmin = -1
            for dp2 in (-1, 0, 1):
                for dp1 in (-1, 0, 1):
                    if dp1 == 0 and dp2 == 0: 
                        self.nn[iP, 1, 1] = iP
                        continue
                    for dq2 in (-1, 0, 1):
                        if not self.PBC[1]: dq2 = 0
                        for dq1 in (-1, 0, 1):
                            if not self.PBC[0]: dq1 = 0
                            nmQ = nmP + np.array([dp1, dp2]) + dq1*self.simTorus[0] + dq2*self.simTorus[1]
                            iQ = np.where((self.list[:,0] == nmQ[0]) & (self.list[:,1] == nmQ[1]))[0]
                            if len(iQ) > 0:
                                self.nn[iP, 1+dp1, 1+dp2] = iQ[0]

    #   Brioullin zone
        self.b1, self.b2 = reciprocalLatticeVectors(self.A1, self.A2)
        self.B1, self.B2 = reciprocalLatticeVectors(self.a1, self.a2)
        self.BZone = tesselate(self.B1, self.B2)
        d = int(max(np.linalg.norm(self.B1), np.linalg.norm(self.B2))/min(np.linalg.norm(self.b1), np.linalg.norm(self.b2)))
        self.klist = pointsWithin(d, self.BZone, self.b1, self.b2, (1E-8*(self.b1/5+self.b2/3)))
        self.klist_xy = np.dot(self.klist, [self.b1, self.b2])

        if len(self.klist) != self.N:
            print("Error: number of sites mismatch #list, #kxlist = %d, %d"%(self.N, len(self.klist)))

        nmin, nmax = np.amin(self.klist[:,0]), np.amax(self.klist[:,0])
        mmin, mmax = np.amin(self.klist[:,1]), np.amax(self.klist[:,1])
        self.invlistkmin = np.array([nmin, mmin])
        self.invklist = -np.ones((np.abs(nmax-nmin)+1, np.abs(mmax-mmin)+1), dtype=int)
        i = 0
        for i in range(len(self.klist)):
            nm = self.klist[i] - self.invlistkmin
            self.invklist[nm[0], nm[1]] = i
            i += 1

                        
    #-----------------------------------------------------------------------------------------
    def int2str(self, n, s, first=False):
        if n > 0:
            if first:
                return str(n) + s
            else:
                return '+' + str(n) + s
        elif n < 0:
            return  str(n) + s
        else:
            return ''
        
    #-----------------------------------------------------------------------------------------
    def __str__(self):
        s  = "a1 = " + str(self.a1) + "\n"
        s += "a2 = " + str(self.a2) + "\n"
        s += "A1 = " + str(self.A1) + "\n"
        s += "A2 = " + str(self.A2) + "\n"
        s += "b1 = " + str(self.b1) + "\n"
        s += "b2 = " + str(self.b2) + "\n"
        s += "B1 = " + str(self.B1) + "\n"
        s += "B2 = " + str(self.B2) + "\n"
        s += "STcell = " + str(self.STcell) + "\n"
        s += "BZone = " + str(self.BZone) + "\n"
        s += "PBC = " + str(self.PBC) + "\n"
        s += "N = " + str(self.N) + "\n"

        return s
    
    #-----------------------------------------------------------------------------------------
    def plot(self, filename=None):
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16,7))
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle(r'$N = %d$,   ${\bf A}_1 = %s %s$,   ${\bf A}_2 = %s %s$'%
            (self.N, self.int2str(self.simTorus[0,0], ' {\\bf a}_1', first=True), self.int2str(self.simTorus[0,1], ' {\\bf a}_2', first=(True if self.simTorus[0,0] == 0 else False)), 
                     self.int2str(self.simTorus[1,0], ' {\\bf a}_1', first=True), self.int2str(self.simTorus[1,1], ' {\\bf a}_2', first=(True if self.simTorus[1,0] == 0 else False))), fontsize=16)
        self.plotPanel(ax[0], 'a', 'A', self.a1, self.a2, self.A1, self.A2, self.WScell, np.dot(self.list, [self.a1, self.a2]), color='#4444aa', labelx=r'$x$', labely=r'$y$')
        self.plotPanel(ax[1], 'b', 'B', self.b1, self.b2, self.B1, self.B2, self.BZone, np.dot(self.klist, [self.b1, self.b2]), color='#229922', labelx=r'$k_x$', labely=r'$k_y$')
        if filename:
            plt.savefig(filename)

    #-----------------------------------------------------------------------------------------
    def plotPanel(self, ax, s1, s2, a1, a2, A1, A2, cell, xy, color='#4444aa', labelx=r'$x$', labely=r'$y$'):
        ax.set_aspect(1)
        ax.set_xlabel(labelx, fontsize=16)
        ax.set_ylabel(labely, fontsize=16)
        ax.add_patch(patches.Polygon(cell, color=color, closed=True, fill=True, alpha=0.1))
        ax.scatter(xy[:,0], xy[:,1], color=color)
        
        hw = max(np.linalg.norm(A1), np.linalg.norm(A2)) * 0.05
        ax.arrow(0, 0, a1[0], a1[1], length_includes_head=True, head_width=hw, color='#aaaaaa')
        ax.arrow(0, 0, a2[0], a2[1], length_includes_head=True, head_width=hw, color='#aaaaaa')
        ax.text((a1+(a1+a2)*0.1)[0], (a1+(a1+a2)*0.1)[1], r'${\bf %s}_1$'%s1, color='#aaaaaa', ha='center', va='center', fontsize=12)
        ax.text((a2+(a1+a2)*0.1)[0], (a2+(a1+a2)*0.1)[1], r'${\bf %s}_2$'%s1, color='#aaaaaa', ha='center', va='center', fontsize=12)
        ax.arrow(0, 0, A1[0], A1[1], length_includes_head=True, head_width=hw, color=color)
        ax.arrow(0, 0, A2[0], A2[1], length_includes_head=True, head_width=hw, color=color)
        ax.text((A1+(a1+a2)*0.1)[0], (A1+(a1+a2)*0.1)[1], r'${\bf %s}_1$'%s2, color=color, ha='center', va='center', fontsize=12)
        ax.text((A2+(a1+a2)*0.1)[0], (A2+(a1+a2)*0.1)[1], r'${\bf %s}_2$'%s2, color=color, ha='center', va='center', fontsize=12)
        
        v = np.array([[0,0], A1, A1+A2, A2])
        ax.plot(v[:,0], v[:,1], color=color, zorder=-100, alpha=0.5)

        L = 4*int(max(np.linalg.norm(A1), np.linalg.norm(A2)))
        p = np.empty((0,2))
        for j in np.arange(-L,L):
            for i in np.arange(-L,L):
                p = np.vstack(( p, i*a1 + j*a2 ))
        ax.scatter(p[:,0], p[:,1], marker='o', linewidth=0.5, color='#cccccc', zorder=-200)

        v = np.vstack((np.array([A1, A2, a1, a2, -a1, -a2]), cell))
        xmin, xmax = 0, 0
        ymin, ymax = 0, 0
        for x,y in v:
            if xmax < x: xmax = x
            if xmin > x: xmin = x
            if ymax < y: ymax = y
            if ymin > y: ymin = y
        ax.set_xlim(xmin*1.2, xmax*1.2)
        ax.set_ylim(ymin*1.2, ymax*1.2)

        for i,p in enumerate(xy):
            ax.text(p[0], p[1], r'$%d$'%i, color='#cccccc', ha='center', va='center', fontsize=4)
        for d2 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                if d1 == 0 and d2 == 0: continue
                for i,p in enumerate(xy):
                    q = p + d1*A1 + d2*A2
                    if q[0] < xmin*1.2 or q[0] > xmax*1.2: continue
                    if q[1] < ymin*1.2 or q[1] > ymax*1.2: continue
                    ax.text(q[0], q[1], r'$%d$'%i, color='#666666', ha='center', va='center', fontsize=4)

        return ax


#-----------------------------------------------------------------------------------------
def reciprocalLatticeVectors(v1, v2): 

    """Returns the reciprocal lattice vectors."""

    return np.transpose(2*np.pi*np.linalg.inv(np.stack((v1, v2), axis=0)))

#-----------------------------------------------------------------------------------------
def tesselate(A1, A2): 

    """Return the vertices of the Wigner Seitz cell, or Brioullin zone in their consecutive
    order."""

#     if np.dot(A1,A2) > 0:
#         vecs = np.array([A1, A2, A1+A2, -A1, -A2, A1-A2, A1], dtype=np.double)/2.
#     else:
#         vecs = np.array([A1, A1+A2, A2, -A1, -A1-A2, -A2, A1], dtype=np.double)/2.
    if np.dot(A1,A2) > 0:
        vecs = np.array([A2, A1, -A2 + A1, -A2, -A1, A2 - A1, A2], dtype=np.double)/2.
    else:
        vecs = np.array([A2, A1 + A2, A1, -A2, -A1 - A2, -A1, A2], dtype=np.double)/2.

    if np.abs(np.dot(A1, A2/np.linalg.norm(A2))) > np.linalg.norm(A2):# or np.abs(np.dot(A2, A1/np.linalg.norm(A1))) > np.linalg.norm(A1):
        vecs = np.array([A2, A2 - A1, -A2, A1 - A2])

    vertices = np.empty((0,2), dtype=np.double)
    for i in range(len(vecs)-1):
        xing = intersect(vecs[i], vecs[i]+cross(vecs[i]), vecs[i+1], vecs[i+1]+cross(vecs[i+1]))
        if i > 0 and np.linalg.norm(xing - vertices[-1]) < 1E-12: continue  # avoid degenerate crossing points
        vertices = np.vstack(( vertices, xing ))

    return vertices

#-----------------------------------------------------------------------------------------
def sortByAngle(list):
    offset = 0.01/3. + 0.03/7.j  # to avoid accidental degenerate angles
    return list[np.argsort(np.angle(list[:,0] + 1j*list[:,1] + offset))]

#-----------------------------------------------------------------------------------------
def intersect(P1, P2, P3, P4): 

    """Return the intersection of two lines, given two points on each line segment.
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection"""

    t = ((P1[0]-P3[0])*(P3[1]-P4[1]) - (P1[1]-P3[1])*(P3[0]-P4[0])) / ((P1[0]-P2[0])*(P3[1]-P4[1]) - (P1[1]-P2[1])*(P3[0]-P4[0]))

    return P1 + t*(P2-P1)

#-----------------------------------------------------------------------------------------
def cross(vec):
    return np.array([-vec[1], vec[0]])

#-----------------------------------------------------------------------------------------
def pointsWithin(L, poly, v1, v2, shift): 

    """Returns all points n * v1 + m * v2, with integers n and m, which lie within/on the 
    boundary around (0,0) set by the vertices in poly, respecting periodic boundary 
    conditions."""

    sites = np.empty((0,2), dtype=int)
    for j in np.arange(-2*L, 2*L):
        for i in np.arange(-2*L, 2*L):
            P = i*v1+ j*v2
            if isWithin(P, poly + shift) > 0:
                sites = np.vstack(( sites, np.array([i, j]) ))

    return sites

#-----------------------------------------------------------------------------------------
def isWithin(P, poly): 

    """Returns +1 if P lies within the polygon defined by the certices in poly; −1 if P 
    lies outside of poly; 0 if P lies on poly."""

    poly = np.vstack(( poly, poly[0] ))
    t = -1
    for i in range(len(poly)-1):
        t = t * crossProductTest(P, poly[i], poly[i+1])
#        if t == 0: break

    return t

#-----------------------------------------------------------------------------------------
def crossProductTest(A, B, C):

    """Returns −1, if the line emenating from A to the right cuts the line spanned by [BC]; 
    0 is A lies on [BC]; +1 else."""

    if np.abs(A[1]-B[1]) < 1E-12 and np.abs(B[1]-C[1]) < 1E-12:
        if B[0] <= A[0] <= C[0] or C[0] <= A[0] <= B[0]:
            return 0
        else:
            return 1
    if np.abs(A[1]-B[1]) < 1E-12 and np.abs(A[0]-B[0]) < 1E-12: return 0
    if B[1] > C[1]:
        tmp = B
        B = C
        C = tmp
    if A[1] <= B[1] or A[1] > C[1]: return 1

    Delta = (B[0]-A[0]) * (C[1]-A[1]) - (B[1]-A[1]) * (C[0]-A[0])
    if Delta > 0:
        return -1
    elif Delta < 0:
        return 1
    else:
        return  0

#-----------------------------------------------------------------------------------------
def createCommon(latt, lattName):

    """Returns the site and bond lists for common lattices, based on the provided Bravais lattice.
    
       N                             the number of sites (unit cells * nOrb)
       nOrb                          the number of orbitals per unit cell
       list[site,:]                  the unit cell and orbital
       invlist[unit cell, orbital]   the site index for the provided orbital in unit cell
       nb                            the number of bonds
       bnd[bond,:]                   the two sites belonging to a bond
    """

    if lattName == 'chain':
        nOrb = 1
        N = latt.N * nOrb
        
        list = -np.ones((latt.N*nOrb, 2), dtype=int)
        invlist = -np.ones((latt.N, nOrb), dtype=int)
        
        nc = 0
        for nu in range(latt.N):
            for no in range(nOrb):
                list[nc, 0] = nu
                list[nc, 1] = no
                invlist[nu, no] = nc
                nc = nc + 1

        nb = latt.N
        bnd = -np.ones((nb,2), dtype=int)
        for nu in range(latt.N):
            bnd[nu] = [invlist[nu, 0], invlist[latt.nn[nu, 1+1, 1+0], 0]]

    elif lattName == 'square':
        nOrb = 1
        N = latt.N * nOrb
        nbUC = 2
        
        list = -np.ones((latt.N*nOrb, 2), dtype=int)
        invlist = -np.ones((latt.N, nOrb), dtype=int)
        
        nc = 0
        for nu in range(latt.N):
            for no in range(nOrb):
                list[nc, 0] = nu
                list[nc, 1] = no
                invlist[nu, no] = nc
                nc = nc + 1

        nb = nbUC*latt.N
        bnd = -np.ones((nb,2), dtype=int)
        for nu in range(latt.N):
            bnd[nu*nbUC  ] = [invlist[nu, 0], invlist[latt.nn[nu, 1+1, 1+0], 0]]
            bnd[nu*nbUC+1] = [invlist[nu, 0], invlist[latt.nn[nu, 1+0, 1+1], 0]]

    elif lattName == 'honeycomb':
        nOrb = 2
        N = latt.N * nOrb
        nbUC = 3
        
        list = -np.ones((latt.N*nOrb, 2), dtype=int)
        invlist = -np.ones((latt.N, nOrb), dtype=int)
        
        nc = 0
        for nu in range(latt.N):
            for no in range(nOrb):
                list[nc, 0] = nu
                list[nc, 1] = no
                invlist[nu, no] = nc
                nc = nc + 1
 
        nb = nbUC*latt.N
        bnd = -np.ones((nb,2), dtype=int)
        for nu in range(latt.N):
            bnd[nu*nbUC  ] = [invlist[nu, 0], invlist[latt.nn[nu, 1+0, 1+0], 1]]
            bnd[nu*nbUC+1] = [invlist[nu, 0], invlist[latt.nn[nu, 1+1, 1-1], 1]]
            bnd[nu*nbUC+2] = [invlist[nu, 0], invlist[latt.nn[nu, 1+0, 1-1], 1]]

    elif lattName == 'triangular':
        nOrb = 1
        N = latt.N * nOrb
        nbUC = 3
        
        list = -np.ones((latt.N*nOrb, 2), dtype=int)
        invlist = -np.ones((latt.N, nOrb), dtype=int)
        
        nc = 0
        for nu in range(latt.N):
            for no in range(nOrb):
                list[nc, 0] = nu
                list[nc, 1] = no
                invlist[nu, no] = nc
                nc = nc + 1

        nb = nbUC*latt.N
        bnd = -np.ones((nb,2), dtype=int)
        for nu in range(latt.N):
            bnd[nu*nbUC  ] = [invlist[nu, 0], invlist[latt.nn[nu, 1+1, 1+0], 0]]
            bnd[nu*nbUC+1] = [invlist[nu, 0], invlist[latt.nn[nu, 1+1, 1-1], 0]]
            bnd[nu*nbUC+2] = [invlist[nu, 0], invlist[latt.nn[nu, 1+0, 1-1], 0]]
            
    else:
        print('No lattice definitions for <%s> could not found.'%lattName)

    return N, nOrb, list, invlist, nb, bnd

#=========================================================================================
if (__name__ == '__main__'):
    print('This is the 2Dlattice module, by thomas.lang@uibk.ac.at, version %s\n'%__version__)

