import qinfFun as fn
import grdFunsVB as inpSc
import numpy as np
import scipy.stats as scstat
import scipy.linalg as sclin
import time as time
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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
