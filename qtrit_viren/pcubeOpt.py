
# coding: utf-8

# In[3]:

#Aim: For a pcubed qubit input output channel compute the maximized coherent information 
#     using scipy local optimization techniques.
#     Print files containing lists
#           1. (b, c, Q(1))
#           2. (b, c, rho_opt)
#Author: Vikesh Siddhu,
#        Department of Physics,
#        Carnegie Mellon University,
#        Pittsburgh PA, USA
#Date : 25th Sept'17


# In[4]:

import qinfFun as fn
import grdFuns as inpSc

import numpy as np
import scipy.stats as scstat
import scipy.linalg as sclin
import time as time

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from tabulate import tabulate
from astropy.table import Table


# In[5]:

#from http://www.kosbie.net/cmu/fall-14/15-112/notes/file-and-web-io.py
def writeFile(filename, contents, mode="wt"):
    # wt = "write text"
    with open(filename, mode) as fout:
        fout.write(contents)

def readFile(filename, mode="rt"):
    # rt = "read text"
    with open(filename, mode) as fin:
        return fin.read()


# In[6]:

tStart = time.time()
qlst = []
cVals = []
bVals = []
rhoOpt = []


# In[7]:

c = -0.5
d = 3
bMin = -c/2.
bMax = -2.*c
totalB = 2
stepB = (bMax - bMin)/totalB


# In[8]:

for run in xrange(0,totalB):
    b = bMin + stepB*run
    prm = (b, c)
    
    #Minimize the entropy difference S(C) - S(B) to obtain a maximum for 
    #S(B) - S(C)
    no = int(d*d)
    bounds = [(-1,1)]*no
    resG = differential_evolution(inpSc.entBias, bounds, args = prm, popsize = 40, disp = False)

    xOpt = resG.x
    xOpt = xOpt/(np.linalg.norm(xOpt))

    #Refine the global optimization by performing a secong local optimizaiton
    x0 = xOpt

    res = minimize(inpSc.entBias, x0, args = prm, method='BFGS', options={'disp': False})
    xOpt = res.x
    xOpt = xOpt/(np.linalg.norm(xOpt))
    rhoOp = inpSc.getMat(xOpt, d)
    opQ = -res.fun

#Put the Q(1)'s in a list and write them to a file as tables
    qlst += [(b, c, opQ)]

#Put the (b, c, rho_alg, rho_opt) in columns of bVals, cVals... and

    bVals += [b]
    cVals += [c]
    rhoOpt += [rhoOp]


# In[9]:

#Write Stuff
table =  tabulate(qlst, headers = ['b', 'c', 'Q(1)']) + '\n'
filename1 = 'resXTable.txt'
writeFile(filename1, table, mode = 'a')

filename2 = 'resQX'
np.savez(filename2, qlst = qlst)

#Add bVals, cVals, rhoAls, rhoOpt to an astro table, write this table to a file
tab = [bVals, cVals, rhoOpt]
table = Table(tab, names = ['b', 'c', 'rho_Opt'])

table.write('resMX.hdf5', path ='/data')

tEnd = time.time()

print -tStart + tEnd

