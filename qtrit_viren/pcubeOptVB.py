
# coding: utf-8

# In[41]:

# Aim: For a pcubed qubit input output channel compute the maximized coherent information 
#     using scipy local optimization techniques.
#     Outputs files containing dictonary(pickled) of list of nodes, each of which contains the fields:
#         b 
#         c 
#         Q1  
#         Optimized Density 
#         Distance (Shatten 1 Norm)
#         Eigen values  
#         Eigen vectors
        
# Author: Viren Bajaj,
#        Department of Physics,
#        Carnegie Mellon University,
#        Pittsburgh PA, USA
# Adapted from Vikesh Siddhu's file: pcubeOpt
# Date : 29 Nov'17


# In[42]:

import qinfFun as fn
import grdFunsVB as inpSc

import numpy as np
import scipy.stats as scstat
import scipy.linalg as sclin
import time as time
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from tabulate import tabulate
from astropy.table import Table
import cPickle as pickle


# In[43]:

#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
#Easy pickle tutorial: http://www.bogotobogo.com/python/python_serialization_pickle_json.php

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f) 
        #want ASCII format for readability. 
        #Use pickle.HIGHEST_PROTOCOL as third argument in dump for binary format.
        #stay consistent in dumping and loading in same format: binary ('wb','rb')

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[44]:

class Node:
    """
    Calculates and stores the optimized variables for a given (b,c) pair 
    needed for later analysis
    
    """
    def __init__(self,b,c,d=3):
        self.b = b
        self.c = c
        self.d = d
        self.Q1 = 0.
        self.eVals = None
        self.eVecs = None
        self.dist = 0.
        self.rhoOp = None
        
    def __repr__(self):

        return """
            b = {0} \n
            c = {1} \n
            Q1 = {2} \n 
            Optimized Density = {3} \n
            distance = {4} \n 
            eigen values = {5} \n 
            eigen vectors = {6} \n
            """.format(self.b,self.c,self.Q1,self.rhoOp,self.dist, self.eVals, self.eVecs)   

    def optimize(self):
        """
        optimize() --> None
        
        Calculates rhoOp and Q1 and stores it.
        """
        prm = (self.b,self.c)
        d = self.d
        no = int(d*d)
        bounds = [(-1,1)]*no
        resG = differential_evolution(inpSc.entBias, bounds, args = prm, popsize = 40, disp = False)

        xOpt = resG.x
        xOpt = xOpt/(np.linalg.norm(xOpt))

        #Refine the global optimization by performing a second local optimizaiton
        x0 = xOpt

        res = minimize(inpSc.entBias, x0, args = prm, method='BFGS', options={'disp': False})
        xOpt = res.x
        xOpt = xOpt/(np.linalg.norm(xOpt))
        self.rhoOp = inpSc.getMat(xOpt, d)
        self.Q1 = -res.fun
    
    @staticmethod
    def S1(A,B):    
        """
        S1(2-d np array, 2-d np array) --> float

        Returns Shatten 1 distance (Trace Norm) between matrices A,B 
        which is equal to Trace(sqrt(A-B))
        (see https://quantiki.org/wiki/trace-norm)

        Arguments:
            A: 2-d np array 
            B: 2-d np array

        Returns:
            S1 distance: float 

        Quirk: A,B must be hermition so sqrt(A-B) is defined (????????????).
        """
        C = np.subtract(A,B)
        s = np.linalg.svd(C)[1]
        return (np.sum(s))
    
    def calcEVals(self):
        """
        calcEVals() --> None
        Calculates and stores eigen values of rhoOp.
        
        """
        self.eVals,self.eVecs = np.linalg.eigh(self.rhoOp)
   
    def calcDist(self):
        """
        calcDist() --> None
        Calculates and stores the Shatten 1 distance(Trace Norm) between rhoOp and symmetric rhoOp(????????).
        
        """
        rhoOp = self.rhoOp
        s = np.array([[1,0,0],[0,-1,0],[0,0,1]])
        sAdj = s.conj().T 
        symRhoOp = np.dot(s,np.dot(rhoOp,sAdj))
        self.dist = Node.S1(rhoOp, symRhoOp)
        
        
     
        
    


# In[45]:

def calcBRange(c,n=10):
    """
    calcBRange(float, *int) --> 1-d numpy array of length n(=10)
    
    Calculates the range of b values for a given C such that D is P.S.D
    
    Arguments:
        c           : float representing value of c
        n(optional) : integer representing number of valid b values required for the given c
    
    Returns:
        1-d numpy array with n valid b values equally spaced between its range.
        Here 'valid' implies values of b,c such that D is positive semidefinite. 
    
    """
    if c<=0:
        bMin = c
        bMax = -c/2.
    elif c <= .5 and c > 0:
        bMin = -c/2.
        bMax = c
    elif c >.5:
        bMin = -c/2.
        bMax = .5
    return np.linspace(bMin,bMax,n)
    


# In[46]:

totTStart = time.time()
printT = 0 
try:
    d = load_obj("optVarDict")
except:
    d = {}
cRange= np.linspace(-.4,.9,13)
for c in cRange:
    cTStart = time.time()
    bRange = calcBRange(c)
    cStr = str(round(c,4))
    key = hash(cStr)
    bListTemp = []
    for b in bRange:
        node = Node(b,c)
        node.optimize()
        bListTemp.append(node)
    d[key] = bListTemp
    cTEnd = time.time()
    printTStart = time.time()
    print "optimized for c = ",c, "time taken = ", cTEnd-cTStart
    printT += (time.time()-printTStart)

totT = time.time() - totTStart
totOptT = totT - printT 
print totOptT, printT


# In[47]:

if d != {}:
    save_obj(d,"optVarDict")
else:
    print "dict empty"


# In[48]:

d = load_obj("optVarDict")


# In[49]:

for bList in d.values():
    for node in bList:
        node.calcDist()
        node.calcEVals()


# In[55]:

# def f(self):
#     print self.b
# Node.check = f

# for node in d.values():
#     node.check()


# In[50]:

import h5py
import matplotlib.pyplot as plt


# In[51]:

keys = sorted(d.keys())
temp = np.linspace(-.4,.9,13)
b = sorted([hash(str(round(a,4))) for a in temp ])
print len(b)
print len(keys)


# In[53]:

cRange = np.linspace(-.4,.9,13)
for c in cRange:
    c = round(c,4)
    cStr = str(c)
    key = hash(cStr)
    cNodes = d[key]
    bVals = [node.b for node in cNodes]
    dVals = [node.dist for node in cNodes]
    #begin figure and plot
    fig = plt.figure()
    bRange = calcBRange(c)
    (bMin,bMax) = bRange[0]-1, bRange[-1]+1
    #make x,y axis
    # plt.plot((-1.,0.5),(0,0),'k--')
    # plt.plot((0,0),(-1.0,1.0),'k--')
    plt.xlabel('b', fontsize=15)
    plt.ylabel('dist', fontsize=15)
    plt.xlim(bMin,bMax)
    dMax = max(dVals)+1
    plt.ylim(0,dMax)
    plt.plot(bVals,dVals,"o")
    #title
    title = 'dist vs b for c = {0}'.format(cStr)
    print title              
    fig.suptitle(title, fontsize=20)
    # plt.show()

    title1 = "dVsb1_{}.png".format(cStr)
    fig.savefig(title1)


# In[ ]:




# In[93]:

def calcKey(c): 
    return hash(str(round(c,4)))
    


# In[128]:

c = -0.3929
key = calcKey(c)
bList = d[key]
print [node.dist for node in bList]
print sorted([node.b for node in bList])
print [node.rhoOp for node in bList]


# In[127]:

c = .5714
key = calcKey(c)
bList = d[key]
print [node.dist for node in bList]
print sorted([node.b for node in bList])
print [node.rhoOp for node in bList]


# In[ ]:



