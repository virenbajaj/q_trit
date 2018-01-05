
# coding: utf-8

# In[ ]:

import qinfFun as fn
import numpy as np
import scipy.linalg as sclin


# In[ ]:

def getMat(x,d):
    """
    getMax(1-d numpy array, int) --> 2-d numpy array
    
    Takes as input a vector and dimension of matrix, returns the
    PSD matrix constructed by putting the vector on an upper triangular matrix A 
    and returning A.A^\dag
    
    Arguments:
        x :  A 1-d numpy array of length d(d+1)/2
        d :  Dimension of the underlying hilbert space
        
    Returns
        mt : A 2-d numpy array of shape (d,d)
        
    Quirk: Does not check whether the dimesion of x and d are consistent

    """
    qqupsR = np.zeros((d,d))
    qqupsC = np.zeros((d,d))
    
    indR = np.triu_indices(d)
    indC = np.triu_indices(d, k = 1)
    
    r = (d*d + d)/2
    real = x[:r]
    ima = x[r:]

    qqupsR[indR] = real
    qqupsC[indC] = ima

    qqups = qqupsR + 1j*qqupsC

    mt = qqups.dot(qqups.conj().T)
    return mt


# In[ ]:

#Helper function for nnba, takes parameters prm and returns a matrix whose
#columns are vectors that constitute the gram matrix [[1,prm,prm],[prm,1,prm],[prm,prm,1]]
#Tested: YES
def gramVec(prm):
    """
    gramVec(float) --> 2-d numpy array
    
    Helper function for nnba the channel super-operator function for
    qutrit pcubed channel
    
    Arguments
        prm : A float that defined the equal off
    
    Returns:
        vecMat : A 2-d numpy array, whose columns represent vectors that give
                 the equal off-diagonal gram matrix. See eq(60) in
                 https://arxiv.org/pdf/1511.05532.pdf
    """
    vecMat = np.array([[np.sqrt((1. + 2*prm)/3.), np.sqrt((1. + 2*prm)/3.), np.sqrt((1. + 2*prm)/3.)],
                        [-np.sqrt((1. - prm)/2.), np.sqrt((1. - prm)/2.), 0.],
                        [-np.sqrt((1. - prm)/6.), -np.sqrt((1. - prm)/6.), np.sqrt(2*(1. - prm)/3.)]]
                )
    return vecMat


# In[ ]:

#Takes as input, the vector of values that defines the input density operator
#and the parameters that define the channel, returns density operator on the b and c side
def nnba(x, prm):
    """
    nnba(1-d numpy array, list) --> (2-d numpy array, 2-d numpy array)
    
    Quantum Channel function, takes a vector and gives the
    direct and complementary channel output for the qutrit pcubed channel
    see eq. (63) in https://arxiv.org/pdf/1511.05532.pdf
    
    Arguments
        x   : A 1-d numpy array used to representing a density operator
              using the A.A^dag form
        prm : (b,c) list of two floats representing off-diagonal terms in
                eq (58) of https://arxiv.org/pdf/1511.05532.pdf
    Returns
        (rhoB, rhoC) where rhoB is the direct channel output and rhoC is
        the complementary
    
    """
    (b,c) = prm
    
    #The Gram matrices B, C and A
    bbq = np.array([[1., b, b], [b, 1., b], [b, b, 1.]])
    ccq = np.array([[1., c, c], [c, 1., c], [c, c, 1.]])
    aaq = np.multiply(bbq,ccq)
    
    (row, col) = np.shape(aaq)
    rho = getMat(x,row)

    #The Gram vectors for B, C and A gram matrices
    mmup = gramVec(b)
    nnup = gramVec(c)
    llup = gramVec(c*b)

    #Inverse of the gram vector matrix
    lbup = sclin.inv(llup)
         
    #Output matrices based on the pcubed formalism
    mtB     = np.multiply(ccq, lbup.dot(rho).dot(lbup.conj().T))  
    rhoB    = mmup.dot(mtB).dot(mmup.conj().T)           
 
    mtC     = np.multiply(bbq, lbup.dot(rho).dot(lbup.conj().T))  
    rhoC    = nnup.dot(mtC).dot(nnup.conj().T)           
    return (rhoB, rhoC)


# In[ ]:

def entOut(x, prm):
    """
    entOut(1-d numpy array, list of length 2) --> list of length 2
    
    Takes as input, the vector of values that defines the input density operator
    and the parameters that define the channel, returns the entropy on the b and c side 
    
    Arguments:
        x   : A 1-d numpy array used to representing a density operator
              using the A.A^dag form
        prm : (b,c) list of two floats representing off-diagonal terms in
                eq (58) of https://arxiv.org/pdf/1511.05532.pdf
    
    Returns:
        (SB, SC) the entropy on the B and C side of the channel
    """
    (rhoB, rhoC) = nnba(x, prm)
    entb = fn.entroVon(rhoB)
    entc = fn.entroVon(rhoC)
    return (entb, entc)


# In[ ]:

def entBias(x, *prm):
    """
    entBias(1-d numpy array, list of length 2) --> float
    
    Takes as input, the vector of values that define the input density operator
    and parameters that defines the channels, returns the entropy bias
    
    Arguments:
        x   : A 1-d numpy array used to representing a density operator
              using the A.A^dag form
        prm : (b,c) list of two floats representing off-diagonal terms in
                eq (58) of https://arxiv.org/pdf/1511.05532.pdf
    
    Returns:
        S(C) - S(B) the entropy difference 
    """
    x = x/(np.linalg.norm(x))
    (entb, entc) = entOut(x, prm)
    return entc - entb

