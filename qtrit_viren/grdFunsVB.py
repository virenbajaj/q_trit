
# coding: utf-8

# In[2]:

# Taken from grdFuns written by Vikesh Siddhu and modified for new analysis. 
# Modified functions:
#     1. gramVec


# In[3]:

import qinfFun as fn
import numpy as np
import scipy.linalg as sclin


# In[4]:

def getMat(x,d):
    """
    getMat(1-d numpy array, int) --> 2-d numpy array
    
    Takes as input a vector and dimension of matrix, returns the
    PSD matrix constructed by putting the vector on an upper triangular matrix A 
    and returning A.A^\dag
    
    Arguments:
        x :  A 1-d numpy array of length d*d
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


# In[5]:

#Helper function for nnba, takes parameters float 'a' and boolean 'new' and returns a matrix whose
#columns are vectors that constitute the gram matrix, i.e. a matrix whose inner product gives the gram matrix: 
#[[1,a,a],[a,1,a],[a,a,1]] when new == False
#[[1,a,a],[a,1,-a],[a,-a,1]] when new == True
#Tested: YES
def gramVec(a,new=False):
    """
    gramVec(float,bool) --> 2-d numpy array
    
    Helper function for nnba the channel super-operator function for
    qutrit pcubed channel
    
    Arguments
        a   : A float that defines the off diagonal elements of the gram matrix
        new : A bool that chooses the correct basis based on the gram matrix to be produced 
    
    Returns:
        vecMat : A 2-d numpy array, whose columns represent vectors that give the
                 1. gram matrix with off-diagonal elements equal to a when new == False
                 2. gram matrix in which all off-diagonal elements are equal to a except 
                    that A_23 and A_32 = -a, when new == True.
                 
                 See eq(60) in https://arxiv.org/pdf/1511.05532.pdf 
    """
    if new:
        vecMatNew = np.array(
            [[a*np.sqrt(2./(1. - a))           , np.sqrt((1 - a)/2.),   np.sqrt((1 - a)/2.)],
             [0.                               , np.sqrt((1 + a)/2.), - np.sqrt((1 + a)/2.)],
             [np.sqrt((1. - a -2*a*a)/(1. - a)), 0.               , 0.                 ]]
                    )
        return vecMatNew
        
    else:
        vecMat = np.array(
            [[ np.sqrt((1. + 2*a)/3.),  np.sqrt((1. + 2*a)/3.), np.sqrt((1. + 2*a)/3.)],
             [-np.sqrt((1. - a)/2.)  ,  np.sqrt((1. - a)/2.)  , 0.                    ],
             [-np.sqrt((1. - a)/6.)  , -np.sqrt((1. - a)/6.)  , np.sqrt(2*(1. - a)/3.)]]
                )
        return vecMat
        

# #tests
# L = gramVec(.125,True)
# Ld = L.conj().T
# #print np.linalg.eigh(np.dot(Ld, L))
# print np.dot(Ld, L) #right
# #print np.dot(L, Ld) #wrong


# In[7]:

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
    bbq = np.array([[1., b, b], [b, 1., -b], [b, -b, 1.]])
    ccq = np.array([[1., c, c], [c, 1., c], [c, c, 1.]])
    aaq = np.multiply(bbq,ccq)
    
    (row, col) = np.shape(aaq)
    rho = getMat(x,row)

    #The Gram vectors for B, C and A gram matrices
    mmup = gramVec(b,new=True)
    nnup = gramVec(c)
    llup = gramVec(c*b,new=True)

    #Inverse of the gram vector matrix
    lbup = sclin.inv(llup)
    
         
    #Output matrices based on the pcubed formalism
    mtB     = np.multiply(ccq, lbup.dot(rho).dot(lbup.conj().T))  
    rhoB    = mmup.dot(mtB).dot(mmup.conj().T)           
 
    mtC     = np.multiply(bbq, lbup.dot(rho).dot(lbup.conj().T))  
    rhoC    = nnup.dot(mtC).dot(nnup.conj().T)           
    return (rhoB, rhoC)


# In[8]:

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


# In[9]:

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

