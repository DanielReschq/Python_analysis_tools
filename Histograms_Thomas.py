#!/usr/bin/env python3
import FS3
import numpy as np
import scipy as sp
import scipy.optimize
import pickle
import sys, getopt


#-----------------------------------------------------------------------------------------
def load(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)
    
#-----------------------------------------------------------------------------------------
def save(obj, filename):
    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid, pickle.HIGHEST_PROTOCOL)

# %%
class f0:
    def __init__(self):
        self.vars = ['gc', 'nu']   # define the variables used in your scaling function
        self.polyOrder = 0         # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1
    
    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var['a']  = params[len(self.vars):self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var['a'])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var['a'][n] * ((g-var['gc']) * L**(1./var['nu']))**n
        return f
    
class f1:
    def __init__(self):
        self.vars = ['gc', 'nu', 'c', 'omega']   # define the variables used in your scaling function
        self.polyOrder = 0         # the polyomial expansion order in your scaling function

    def nparams(self):
        return len(self.vars) + np.sum(self.polyOrder) + 1
    
    def unpack(self, params):
        var = {}
        for name in list(self.vars):
            var[name] = params[self.vars.index(name)]
        var['a']  = params[len(self.vars):self.nparams() + 1]
        return var

    def func(self, g, L, params):
        var = self.unpack(params)
        nmax = len(var['a'])

        f = np.zeros_like(g)
        for n in range(nmax):
            f += var['a'][n] * ((g-var['gc']) * L**(1./var['nu']))**n
        return f * (1 + var['c'] * L**(-var['omega']))


# %%
def resample(data):
    dataSample = {}
    for L in np.sort(list(data),axis=0):
        n = len(data[L])
        dataSample[L] = np.zeros((n,2))
        dataSample[L][:,0] = data[L][:,0]
        dataSample[L][:,1] = data[L][:,1] + np.random.randn(n) * data[L][:,2]
    return dataSample

# %%

if __name__ == "__main__":
    alpha = float(sys.argv[1])
    print(alpha)
    order = int(sys.argv[2])
    print(order)
    corr = int(sys.argv[3])
    print(corr)

    import warnings
    warnings.filterwarnings('ignore')

    nsample = 10000

    if alpha == 4.0: 
        grange = [2.935,2.948] ; g0 = np.mean(grange); nu0 = 0.73
    if alpha == 3.5: 
        grange = [3.1775,3.1885] ; g0 = np.mean(grange); nu0 = 0.73
    if alpha == 3.0: 
        grange = [3.5885,3.6015] ; g0 = np.mean(grange); nu0 = 0.73
    if alpha == 2.5: 
        grange = [4.3225,4.335] ; g0 = np.mean(grange); nu0 = 0.73
    if alpha == 2.0: 
        grange = [5.788,5.8005] ; g0 = np.mean(grange); nu0 = 0.73
    if alpha == 1.5: 
        grange = [9.955,10.001] ; g0 = np.mean(grange); nu0 = 0.73
    data = FS3.load('../DataFS3/ZeroTemp/alpha%.2f/binder.dat'%alpha, LIndex=0, xIndex=1, yIndex=2, dyIndex=3)
    Llist = np.sort(list(data))
    FS3.setStyle(Llist=Llist)
    data = FS3.select(data, gRange=grange, LRange=[64,])

    if not corr:
        resList = np.empty((0,3))
        for i in range(nsample):
            fitFunction = f0()
            fitFunction.polyOrder = order
            params0 = np.random.randn(fitFunction.nparams())
            params0[0] = g0+np.random.randn()*0.001
            params0[1] = nu0+np.random.randn()*0.001
            
            res = FS3.fit_minimize(resample(data), fitFunction, params=params0, **{'method': 'BFGS'})
            params, dparams,redChi2, mesg, ierr = res
            var = fitFunction.unpack(params)
            if (redChi2 < 1000): resList = np.vstack((resList, np.array([var['gc'], var['nu'], redChi2])))
        save(resList, 'fit_alpha%3.2f_nu_O%d.pkl'%(alpha, fitFunction.polyOrder))
    else:
        resList = np.empty((0,3))
        for i in range(nsample):
            fitFunction = f1()
            fitFunction.polyOrder = order
            params0 = np.random.randn(fitFunction.nparams())*0.0001

            match (alpha):
                case 1.5:
                    params0[0] = 9.98651 +np.random.randn()*0.0001
                    params0[1] = 0.7495 +np.random.randn()*0.0001
                    params0[2] = 8 +np.random.randn()*0.0001
                    params0[3] = 0.98 +np.random.randn()*0.0001
                    params0[4] = 0.092 +np.random.randn()*0.0001
                    params0[5] = -0.004 +np.random.randn()*0.0001
                case 2.0:
                    params0[0] =  5.79 +np.random.randn()*0.0001
                    params0[1] =  0.73 +np.random.randn()*0.0001
                    params0[2] = -0.99 +np.random.randn()*0.0001
                    params0[3] = -0.00 +np.random.randn()*0.0001
                    params0[4] = 47.58 +np.random.randn()*0.0001
                    params0[5] = -3.03 +np.random.randn()*0.0001
                    params0[6] =  0.14 +np.random.randn()*0.0001
                case 2.5:
                    params0[0] = 4.33 +np.random.randn()*0.001
                    params0[1] = 0.72 +np.random.randn()*0.001
                    params0[2] = 1.7 +np.random.randn()*0.001
                    params0[3] =  0.8 +np.random.randn()*0.001
                    params0[4] =  0.32 +np.random.randn()*0.001
                case 3.0:
                    params0[0] =  3.59 +np.random.randn()*0.001 
                    params0[1] =  0.70 +np.random.randn()*0.001 
                    params0[2] = -0.57 +np.random.randn()*0.001 
                    params0[3] = -0.01 +np.random.randn()*0.001 
                    params0[4] =  1.20 +np.random.randn()*0.001 
                    params0[5] = -0.06 +np.random.randn()*0.001 
                case 3.5:
                    params0[0] =  3.18 +np.random.randn()*0.001 
                    params0[1] =  0.71 +np.random.randn()*0.001 
                    params0[2] = -0.86 +np.random.randn()*0.001 
                    params0[3] =  0.001 +np.random.randn()*0.001 
                    params0[4] =  4.20 +np.random.randn()*0.001 
                    params0[5] = -0.3 +np.random.randn()*0.001 
                case 4.0:
                    params0[0] =  2.94 +np.random.randn()*0.001 
                    params0[1] =  0.70 +np.random.randn()*0.001 
                    params0[2] = -0.97 +np.random.randn()*0.001 
                    params0[3] =  0.00 +np.random.randn()*0.001 
                    params0[4] =  17. +np.random.randn()*0.001 
                    params0[5] = -1. +np.random.randn()*0.001 

            res = FS3.fit_minimize(resample(data), fitFunction, params=params0, **{'method': 'BFGS'})
            params, dparams,redChi2, mesg, ierr = res
            var = fitFunction.unpack(params)
            if (redChi2 < 1000): resList = np.vstack((resList, np.array([var['gc'], var['nu'], redChi2])))
        save(resList, 'fit_alpha%3.2f_nu_corr_O%d.pkl'%(alpha, fitFunction.polyOrder))

    warnings.filterwarnings('default')

    sys.exit(0)
