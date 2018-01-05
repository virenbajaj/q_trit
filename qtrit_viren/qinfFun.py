#Contains useful functions for quantum channel calculations
#Author: Vikesh Siddhu, 
#        Quantum Theory Group,
#        Carnegie Mellon University
#
#Version: 20th Sept'17
#
#List of Functions
#zerOut(mt)             :   Takes in array, kills small entries
#zerOutNum(mt)          :   Takes in number, kills small entries
#dyad(a,b)              :   Dyad |a><b| is returned
#stanBra(size, index)   :   bra of given 'size' with 1 at index position
#stanKet(size, index)   :   ket of given 'size' with 1 at index position
#tenProd(v1,v2)         :   Tensor product of two arrays 
#isRightDim(mt,da,db)   :   Checks whether mt is squre with dim = da.db
#isPSD(mt)              :   Checks whether matrix is PSD or not
#krausComp(krausAB)     :   Kraus Operators for complementary channel
#krausToOp(krausAB)     :   converts kraus operator to Super-Operator as a rank 4 tensor
#opToChoi(superOp)      :   Converts super-operator to choi tensor as rank 4 tensors
#choiTenToMat(choiTen)  :   Converts choi matrix rank 4 choi tensor to choi matrix
#matToTen(mt,da,db)     :   Gives a rank 4 tensor representation (da db da db) of the matrix
#partTr(mt,da,db)       :   Takes a matrix AB and computes its partial trace over B
#choiMatToTen(...)      :   Converts choi matrix to rank 4 tensor 
#fullRankA(...)         :   Takes mt,da,db on AB and returns Rank(Tr(B(mt)) != da
#choiToKrs(..)          :   Takes choi matrix(on system AB) return the kraus operators A -> B channel
#entroVon(mt)           :   Takes in matrix, checks positivity, normalizes eigen values and computes the entropy
#dualMat(mts)           :   Takes list of matrices, computed dual.
#proj(j,k,n)            :   Takes integers j,k,n construct dyad |j><k| in n dim
#hermBasis(n)           :   Takes integer n, gives n^2-1 traceless, hermitian basis
#doLinComb(...)         :   Takes array of numbers and matrices, constructs linear combination
#vecToMat(vec, n)       :   Takes vector and dimension, constructs generalized bloch matrix
#mtInp(m1, m2)          :   Returns Frobenius inner product of matrices m1 and m2 
#matToVec(vec)          :   Takes hermitian, trace 1 matrix and returns bloch vector
#randU(n)               :   Haar random unitary of dimension n
#randPsi(n)             :   Haar random complex vector of dimension n
#randTwoPsi(n)          :   Two Haar random complex vector of dimension n
#sigProd(index)         :   Given tensor product of pauli matrices 
#pauliBasis(n)          :   Gives pauli basis for n qubits

import numpy as np
import scipy.linalg as sclin
import scipy.stats as scstat
import copy

#Takes as input an array and tolerence, nulls out each entry(possibly complex)
#smaller than the tolerence, returns the nulled array
#Tested: YES
def zerOut(array, tol = 1e-15):
    """
    zerOut(array) ->  array

    Arguments:
    array           : A numpy array
    tol[Optional]   : 1e-15
    
    Returns:
    Array with each entry small than tol nulled out
    """
    arr = copy.deepcopy(array)
    for index, val in np.ndenumerate(arr):
        if (np.abs(val.real) < tol):
            arr[index] = (arr[index] - arr[index].conj())/2.
        if (np.abs(val.imag) < tol):
            arr[index] = (arr[index] + arr[index].conj())/2.
    return arr

#Takes as input an number and tolerence, nulls out real and complex
#smaller than the tolerence, returns the nulled number
#Tested: YES
def zerOutNum(num, tol = 1e-15):
    """
    zerOutNum(complex) ->  complex

    Arguments:
    num             : A complex number
    tol[Optional]   : 1e-15
    
    Returns:
    Number with each entry small than tol nulled out
    """
    val = copy.deepcopy(num)
    if (np.abs(val.real) < tol):
        val = (val - val.conj())/2.
    if (np.abs(val.imag) < tol):
        val = (val + val.conj())/2.
    return val

#Takes two vectors a = |a> (m,) and b = |b>(n,) and returns |a><b|
#as an m x n matrix. If any of arguments a or b are matrices, 
#it flattens them out into vectors and then takes a dyad
#Tested: YES
def dyad(a,b):
    """
    dyad(vec1, vec2) ->  matrix

    Arguments:
    a   : (m,) numpy vector
    b   : (n,) numpy vector
    
    Returns:
    (m,n) numpy matrix a * b.H
    """
    return np.outer(a,b.conj())

#Takes size and index of the vector and returns a row
#vector of the given size and 1 at the given index
#Tested: Yes
def stanBra(size, index):
    """
    stanBra(size, index) ->  vector

    Arguments:
    size   : integer
    index  : integer
    
    Returns:
    (1, 'size') numpy array with 1 at 'index' position
    """
    return np.eye(1,size,index)

#takes size and index of the vector and returns a column
#vector of the given size and 1 at the given index
#Tested: Yes
def stanKet(size, index):
    """
    stanKet(size, index) ->  vector

    Arguments:
    size   : integer
    index  : integer
    
    Returns:
    ('size', 1) numpy array with 1 at 'index' position
    """
    return np.transpose(np.eye(1,size,index))

#takes arrays and returns their tensor product
#Tested: Yes
def tenProd(v1,v2):
    """
    tenProd(v1, v2) ->  numpy array

    Arguments:
    v1  : numpy array
    v2  : numpy array
    
    Returns:
    When v1, v2 are both 1-D or 2-D arrays returns tensor product,
    generally it returns the tensor product
    """
    return np.kron(v1,v2)

#Given a matrix and two numbers da, db checks
#whether the matrix is a d (= da*db) dimensional
#square matrix
#Tested: Yes
def isRightDim(mt, da, db):
    """
    isRightDim(matrix, da, db) ->  True/False

    Arguments:
    mt  : numpy array
    da  : integer
    db  : integer
    
    Returns:
    True if matrix is a da*db square matrix, False otherwise.
    """
    d = da*db
    dimCheck = np.shape(mt) == (d,d)
    return dimCheck    

#Takes a matrix, returns whether it is Positive Semi-definite or not
#Tested: YES
def isPSD(mt, tol = 1e-12):
    """
    isPSD(mt, tol) ->  True/False

    Arguments:
    mt              : numpy array
    tol[Optional]   : 1e-12
    
    Returns:
    True if mt is positive semi-definite upto some numerical tolerence.
    False otherwise.
    """
    #Square Matrix
    (row, col) = np.shape(mt)
    sqr = row == col
    if not sqr:
        return False

    #Hermitian
    mtDag = mt.conj().T
    diff = mt - mtDag
    herm = np.linalg.norm(diff) < 10*tol
    if not herm:
        return False

    #Positivity
    mt2 = (mt + mt.conj().T)/2.
    val, vec = sclin.eigh(mt2)
    pos = all(val.real > -tol) & all(val.imag > -tol)
    
    return pos

#Function takes a list of Kraus operators and returns
#returns a list of Kraus operators for the complementary channel
#Tested: Yes
def krausComp(krausAB):
    """
    krausComp(krausAB) ->  list of matrices

    Arguments:
    krausAB : List of numpy 2-D arrays representing kraus operators
              for the input -> direct output
    
    Returns:
    List of 2-D numpy arrays representing Kraus Operators 
    for the input -> complementary output of a quantum channel
    """
    (dc, db, da) = np.shape(krausAB)
    krausAC = []
    for i in xrange(db):
        temp = np.zeros(shape = (dc,da), dtype = complex)
        for k in xrange(dc):
            for l in xrange(da):
                val = krausAB[k][i,l]
                temp[k,l] = val
        krausAC.append(temp)
    return krausAC        

#Takes the Kraus operators and returns a 4-dimensional array
#of the channel superoperator
#Tested: Yes, see krausToOp.py
def krausToOp(krausAB):
    """
    krausToOp(krausAB) ->  4-D array

    Arguments:
    krausAB : List of numpy 2-D arrays representing kraus operators
              for the input -> direct output
    
    Returns:
    A 4-D array representing the input -> direct output superoperator
    in the standard basis with first two indices representing the output
    and the last two representing the input
    """
    (dc, db, da) = np.shape(krausAB)
    temp = np.zeros(shape = (db,db,da,da), dtype = complex)
    it = np.nditer(temp, flags = ['multi_index'])

    while not it.finished:
            (i,j,k,l) = it.multi_index
            val = sum(krausAB[r][i,k]*np.conjugate(krausAB[r][j,l]) for r in xrange(dc))
            temp[i,j,k,l] = val
            it.iternext()
    return temp

#Takes a 4-dimensional superoperator and returns the 
#choi matrix as a 4-dimensional array
#Tested: Yes, see opTpChoi.py

def opToChoi(superOp):
    """
    opToChoi(4-D array superOp) ->  4-D array 

    Arguments:
    superOp : A 4-D numpy array for the a -> b channels
    
    Returns:
    A 4-D representation of the choi matrix representing da*db square
    matrix
    """
    (db, db, da, da) = np.shape(superOp)
    temp = np.zeros(shape = (da,db,da,db), dtype = complex)
    it = np.nditer(superOp, flags = ['multi_index'])

    while not it.finished:
            (k,l,r,s) = it.multi_index
            val = superOp[k,l,r,s]
            temp[r,k,s,l] = val
            it.iternext()
    return temp

#Takes a 4-dimensional tensor representation(da, db, da, db)
#and returns a square matrix of this tensor (da.db x da.db)
#Tested: Yes, see choiTenToMat.py

def choiTenToMat(choiTen):
    """
    choiTenToMat(4-D array) ->  2-D array 

    Arguments:
    choiTen: A 4-D numpy array for the a -> b channels
    
    Returns:
    A 2-D representation of the choi matrix of dimension da*db 
    """
    (da,db,da,db) = np.shape(choiTen)
    d = da*db
    choiMat = np.zeros(shape = (d,d), dtype = complex)
    it = np.nditer(choiTen, flags = ['multi_index'])

    while not it.finished:
        (r,k,s,l) = it.multi_index
        #Kets of Choi Matrix
        kt1 = stanKet(da,r)
        kt2 = stanKet(db,k)
        kt = tenProd(kt1,kt2) 
        #Bra of Choi Matrix 
        br1 = stanBra(da,s)
        br2 = stanBra(db,l)
        br = tenProd(br1,br2) 
        #Operator
        mt = tenProd(kt,br)
        choiMat += choiTen[r,k,s,l]*mt
        it.iternext()

    return choiMat


# Takes a square matrix(dimension d) and dimensions 
# da, db such that d = db*da and returns
# a rank 4 tensor representation (da db da db) of the matrix

def matToTen(mt, da, db):
    """
    matToTen(array, int1, int2) ->  4-D array

    Arguments:
    mt  : A square 2-D numpy array 
    da  : Dimension 1 for representation
    db  : Dimension 2 for representation
    
    Returns:
    A 4-D representation of the matrix, the input array must be
    a square matrix with dimension = da*db
    """
    #Check dimensions match
    dimCheck = isRightDim(mt, da, db) 
    if dimCheck == False:
        raise ValueError('matrix dimension do not match')
    
    mtTen = np.zeros(shape = (da,db,da,db), dtype = complex)
    it = np.nditer(mtTen, flags = ['multi_index'])

    while not it.finished:
        (r,k,s,l) = it.multi_index
        #Kets of Choi Matrix
        kt1 = stanKet(da,r)
        kt2 = stanKet(db,k)
        kt = tenProd(kt1,kt2)
        #Bra of Choi Matrix 
        br1 = stanBra(da,s)
        br2 = stanBra(db,l)
        br = tenProd(br1,br2) 
        #Operator
        bas = tenProd(kt,br)
        basDag = bas.conj().T
        inp = np.matrix.trace( np.dot(basDag, mt) )

        mtTen[r,k,s,l] = inp
        it.iternext()
    
    return mtTen

#Takes a matrix AB and computes its
#partial trace over B

def partTr(mt, da, db):
    """
    partTr(array, int1, int2) ->  2-D array

    Arguments:
    mt: A square 2-D numpy array 
    da: Dimension of space A
    db: Dimension of space B
    
    Returns:
    The partial trace of mt with respect to space B. mt must
    be square with dimension da*db.
    """
    mtTen = matToTen(mt, da, db)
    return np.einsum('ijkj',mtTen)


#Takes a positive semidefinite square matrix(dimension d)
#and dimensions da, db such that d = db*da and returns
#a rank 4 tensor representation (da db da db) of the matrix
#Tested: YES
def choiMatToTen(choiMt, da, db):
    """
    choiMatToTen(array, int1, int2) ->  4-D array

    Arguments:
    choiMt  : A square 2-D numpy array 
    da      : Dimension of space A
    db      : Dimension of space B
    
    Returns:
    A 4 dimensional representation of choimt. choimt must
    be square with dimension da*db and positive semi-definite.
    """
    #Check dimensions match
    dimCheck = isRightDim(choiMt, da, db) 
    if dimCheck == False:
        raise ValueError('Choi matrix dimension do not match')
    
    #Check PSD condition
    psd = isPSD(choiMt)
    if psd == False: 
        raise ValueError('Choi matrix not PSD')
    
    return matToTen(choiMt, da, db) 


#Given a matrix on system AB(mt), dimensions of a and b (da, db)
#compute mA = Tr_B(mt) and if Rank(mt) != da return False, True otherwise

def fullRankA(mt, da, db, tol = 1e-10):
    """
    fullRankA(array, int1, int2) -> bool

    Arguments:
    mt      : A square 2-D numpy array 
    da      : Dimension of space A
    db      : Dimension of space B
    
    Returns:
    True if partial trace with respect to B has full rank,
    False otherwise
    """
    mtA = partTr(mt, da, db)
    return np.linalg.matrix_rank(mtA, tol) == da

#Given a choi matrix(on system AB) and dimensions da, db
#Checks PSD condition 
#Normalized the matrix, such that Tr_B(mat) = I_A
#Return the kraus operators for the A -> B channel

def choiToKrs(choiMt, da, db, tol = 1e-10):
    """
    choiToKrs(array, int1, int2) -> list of int2 x int1 arrays

    Arguments:
    choimt  : A square 2-D numpy array 
    da      : Dimension of space A
    db      : Dimension of space B
    
    Returns:
    Checks if the choiMt is square with dimension da*db and positive
    semi-definite. Returns a list of length equal to the rank(choiMt)
    of Kraus Operators for the a ->b channel.
    """
    
    #Check dimensions match
    dimCheck = isRightDim(choiMt, da, db) 
    if dimCheck == False:
        raise ValueError('Choi matrix dimension do not match')
    
    #Check PSD condition
    psd = isPSD(choiMt)
    if psd == False: 
        raise ValueError('Choi matrix not PSD')

    #Check Full Rank A
    rnkCheck = fullRankA(choiMt, da, db, tol)    
    if rnkCheck == False:
        raise ValueError('Choi Matrix has rank deficient da')

    #Normalize the trace 
    rhoA = partTr(choiMt, da, db)
    nrm = sclin.sqrtm(rhoA)
    nrm = np.linalg.inv(nrm)
    nrm = tenProd(nrm, np.identity(db))
    choiMt = np.dot( np.dot( nrm, choiMt ), nrm )

    #Get Eigensystem
    val, vec = np.linalg.eigh(choiMt)
    val = val.real 
    val[val < tol] = 0
    dc = np.count_nonzero(val)
    cnt = 0

    #Get Kraus Operators
    d = da*db
    krsLst = []

    for i in xrange(len(val)):
        kt = vec[:,i]
        kt = np.reshape(kt,(d,1))
        kt = kt/np.linalg.norm(kt)
        
        if val[i] != 0:
            krs = np.zeros(shape = (db,da), dtype = complex)
            it = np.nditer(krs, flags = ['multi_index'])
            
            while not it.finished:
                (m,l) = it.multi_index
                bm = stanBra(db,m)
                al = stanBra(da,l)
                albm = tenProd(al,bm)
                krs[m,l] = np.dot(albm, np.sqrt(val[i])*kt)[0,0]
                it.iternext()
            
            krsLst += [krs]
    return krsLst            


#Given a  matrix, by default checks positivity, normalizes eigen values(unless all 0) and computes the entropy
#Tested: YES
def entroVon(mat, bs = 2, tol = 1e-10, check = True):
    """
    entroVon(array, int1, float, book) -> float

    Arguments:
    mat             : A square 2-D numpy array 
    bs[Optional]    : The base of log (Default 2)
    tol[Optional]   : Minimum norm of matrix, positivity tolerence(Default 1e-10)
    check[Optional] : If True check for positivity(Default True)
    
    Returns:
    The von-Neumann entropy of 2-d array in base = bs. Nulls out,
    extremely small eigen values.
    """
    if check:
        psd = isPSD(mat, tol)
        if psd == False: 
            raise ValueError('entroVon: matrix not PSD')
    eigs = sclin.eigvalsh(mat).real 
    eigs = zerOut(eigs, tol = tol/100) 
    null =  np.linalg.norm(eigs) < tol
    if null == True :
        raise ValueError('Matrix is Null')
    else:
        return scstat.entropy(eigs, base = bs)
 
#Aim: Takes as input a list of matrices, an optional argument suggesting
#whether the matrices are linearly independent and returns a list of dual matrices
#under the Frobenius Norm
#Tested: YES
def dualMat(mts, lid = False):
    """
    dualMat(list of array, bool) -> list of arrays

    Arguments:
    mts             : A list of square 2-D numpy arrays
    lid[Optional]   : If True assumes matrices are lineraly independent(Default False)
    
    Returns:
    The dual(under Frobenius Norm) matrices to the set of input matrice. 
    If matrices are linearly independent, it uses regular inverse otherwise
    uses(less numerically accurate) pseudo-invers.
    """
    vc = []
    #Each matrix is a ket, place it as a bra making a row of vecRows matrix
    for mt in mts:
        (d,d) = mt.shape
        vcR = np.ndarray.flatten(mt).conj()
        vc += [np.array([vcR])]
    vecRows =  np.vstack(vc)
    #Construct matrix with dual vectors as its columns
    if lid:
        try:
            dualVecCols = np.linalg.inv(vecRows)
        except:
            raise Exception('Cannot find dual matrices')
    else:
        try:
            dualVecCols = np.linalg.pinv(vecRows)
        except:
            raise Exception('Cannot find dual matrices')

    (row, col) = dualVecCols.shape
    mts = []
    #Reshape the columns as matrices of same size as input
    for i in xrange(col):
        mt = dualVecCols[:,i].reshape(d,d)
        mts += [zerOut(mt, tol = 1e-15)]
    return mts

#Takes as input 3 integers j,k,n (0 <= j,k <= n - 1) and returns
#the dyad |j><k|
#Tested: YES
def proj(j,k,n):
    """
    proj(int1, int2, int3) -> n x n numpy array
    
    Arguments:
    j:  Standard Bra index(0 -> n-1)
    k:  Standard Ket index(0 -> n-1)
    n:  dimension of the space

    Returns:
    The dyad |j><k| in the n-dimensional space.
    """
    vj = stanBra(n,j)
    vk = stanBra(n,k)
    return dyad(vj, vk)


#Takes as input 2 integers j, n (expects 1<= j <= n-1) and returns
#a diagonal operator constructed recursively.
#Tested: YES
def getW(j,n):
    """
    getQ(int1, dim) -> n x n numpy array
    Helper Function for hermBasis(n)
    
    Arguments:
    j:  integer index for diagonal operator
    n:  dimension of diagonal operator

    Returns:
    A diagonal matrix operator, constructed recursively. It is 
    the w matrix of Section 2.3 in Generalized Bloch Vector and the Eigenvalues of a Density Matrix
    by Ozols and Mancinska
    """
    if (j >= n-1):
        raise ValueError('Invalid indices')
    if (j == 0 ):
        return proj(0,0,n) - proj(1, 1, n)
    else:
        m1 = getW(j-1, n)*np.sqrt(j*(j+1)/2.)
        m2 = (j+1.)*proj(j,j,n)
        m3 = (j+1.)*proj(j+1, j+1, n)
        return (m1 + m2 - m3)*np.sqrt(2./((j+1.)*(j+2.))) 


#Takes as input a integer n and generates n^2 - 1, traceless,
#hermitian, orthogonal, norm 2 matrices
#Tested: YES
def hermBasis(n):
    """
    hermBasis(int) -> list of n x n numpy array
    
    Arguments:
    n: Dimension of space

    Returns:
    A list of n^2 - 1, traceless, hermitian, orthogonal, norm 2 basis operators.
    They are the lambda matrix of Section 2.3 in 
    Generalized Bloch Vector and the Eigenvalues of a Density Matrix
    by Ozols and Mancinska
    """
    basis = []
    for i in xrange(0, n-1):
        basis += [getW(i,n)]
    for j in xrange(0,n-1):
        for k in xrange(j+1,n):
            u = proj(k,j,n) + proj(j,k,n)
            v = 1j*(proj(k,j,n) - proj(j,k,n))
            basis += [u]
            basis += [v]
    return basis


#Takes in a vector of numbers, a list of matrices and returns a linear combination
#of the matrices.
#Tested: YES
def doLinComb(num, mats):
    """
    diLinComb(array, list of arrays) -> array
    
    Arguments:
    num : A numpy array
    mats: A list of arrays.

    Returns:
    Mutliplies each number in num with corresponding matrix in mats and
    constucts a linear combination.
    """
    if (len(num) != len(mats)):
        raise ValueError('Length of number vector and number of matrices unequal')
        
    (da, da) = mats[0].shape
    rho = 1j*np.zeros(shape = (da, da))
    for i in xrange(len(num)):
        rho += mats[i]*num[i]
    return rho


#Takes as input a real vector, integer n and returns a density operators
#in the generalized bloch sphere representation
#Tested: YES
def vecToMat(vec, n):
    """
    vecToMat(numpy array, int) -> n x n numpy array

    Arguments:
    vec : a vector of length n^2 - 1
    n   : dimension of the space  
    
    Returns:
    n x n matrix in the Generalized Bloch sphere representation.
    See Generalized Bloch Vector and the Eigenvalues of a Density Matrix
    by Ozols and Mancinska
    """
    k = n*n - 1
    if (len(vec) != k):
        raise ValueError('Length of vector not compatible with dimension')
    idMat = np.identity(n)
    basis = hermBasis(n)
    mat = doLinComb(vec, basis)
    mat = np.sqrt(n*(n-1)/2.)*mat
    mat = (mat + idMat)*1./n
    return mat

#Takes as input two matrices and returns their frobenius inner
#product.
#Tested: YES
def mtInp(m1, m2):
    """
    mtInp(numpy array, numpy array) --> array_dType
    
    Arguments:
    m1  : A 2-D numpy array
    m2  : A 2-D numpy array
    m1, m2 must have same number of rows 
    
    Returns:
    The Frobenius inner product of m1 and m2, <m1, m2>
    """
    return np.trace(np.dot(m1.conj().T,m2))

#Takes as input a trace 1, herimitan matrix and returns its bloch vector
#Tested: YES
def matToVec(rho):
    """
    matToVec(2-d numpy array) --> 1-d numpy array

    Arguments:
        rho :   A Hermitian numpy array

    Returns
        vec :   Generalized bloch vector for the Hermitian matrix
    """
    (n,n)   = rho.shape
    basis   = hermBasis(n)
    vec     = np.zeros(len(basis))
    factor  = np.sqrt(n-1.)/np.sqrt(n/2.)
    for i in xrange(len(basis)):
        h   = basis[i]
        vec[i] = mtInp(h,rho).real/factor    
    return vec

#Haar random unitary of dimension n
#Tested: YES
def randU(n):
    """
    randU(int) --> 2-D numpy array
    
    Takes as input a dimension n and returns a random n x n matrix distributed with Haar measure
    See: 1. near eq (35) http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
         2. near eq (5.12) in https://arxiv.org/pdf/math-ph/0609050v2.pdf
    
    Arguments:
        n : An integer
    
    Returns
        q : A random nxn unitary
    """
    z = (np.random.standard_normal(size = (n,n)) + 1j*np.random.standard_normal(size = (n,n)))/np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diag(r)
    ph =np.diag(d/np.absolute(d))
    q = np.dot(q,ph)
    return q


#Haar random complex vector of dimension n
#Tested: YES
def randPsi(n):
    """
    randPsi(int) --> 1-d numpy array

    Takes as input a dimension n and returns a random vector distributed with Haar measure

    Arguments:
        n : An integer
    
    Returns
        q : A random nx1 vector
    """
    psi = np.ones(n)
    psi = psi/np.linalg.norm(psi)
    return np.dot(randU(n),psi)

#Two Haar random complex vector obtained by rotating fixed vector in opposite directions
#of dimension n
#Tested: YES
def randTwoPsi(n):
    """
    randPsi(int) --> (1-d numpy array, 1-d numpy array)

    Takes as input a dimension n and returns two random vector distributed with Haar measure.
    Returns a fixed vector rotated in opposite directions by the same Harr random unitary

    Arguments:
        n : An integer
    
    Returns
        q : random nx1 vector
        r : random nx1 vector
 
    """
    psi = np.ones(n)
    psi = psi/np.linalg.norm(psi)
    uMat = randU(n)
    vMat = uMat.T
    return (np.dot(uMat,psi), np.dot(vMat,psi))
 


#Tested:YES
def sigProd(index):
    """
    sigProd(tuple) --> numpy array
    
    Takes a tuple and converts that into tensor product
    of pauli matrices
    
    Arguments
        index : A tuple of form (i,j,...,l)
    Returns
        mt : sig[i] \ot sig[j] \ot ... \ot \sig[l]
    """
    sig = [np.identity(2), np.array([[0.,1.],[1., 0.]]), 
           np.array([[0.,-1j],[1j, 0.]]), np.array([[1.,0.],[0., -1.]])] 
    
    if type(index) is int:
        return sig[index]
    
    mt = 1.
    for i in index:
        mt = np.kron(mt, sig[i])
    return mt

#Pauli basis for 'n' qubits. Index pauliBasis[i,j,...,l) gives
#sig[i] \ot sig[j] \ot ... \ot \sig[l]
#Tested:YES
def pauliBasis(n):
    """
    pauliBasis(int) --> 2*n-dimensional numpy array
    
    Gives the pauli basis for tensor product of n qubits
    
    Arguments
        n : Number of qubits
    Returns
        2n dimensional pauli basis
    """
    qubits = n
    indices = [4]*qubits
    
    d = 2**qubits
    dims = [d,d]
    
    shapes = indices + dims
    
    basis = 1j*np.zeros(shape = shapes)
    iterator = np.empty(shape = indices)
    
    it = np.nditer(iterator, flags = ['multi_index'])
    while not it.finished:
        basis[it.multi_index] =  sigProd(it.multi_index)
        it.iternext()
    return basis
