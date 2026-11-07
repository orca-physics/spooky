import numpy as np
import scipy
import time
import os

def GMRES(A, b, N_gmres, tol_gmres, gmres_dir: str|None = None, iN: int|None = None, iA: int|None = None):
    """
    Performs Generalized Minimal Residues to find x that approximates the solution to Ax=b.     
    Iterative method that at step n uses Arnoldi iteration to find an orthogonal basis for Krylov subspace
    Assumes initial guess x0 = 0, such that the initial residue is r0 = b.
    If iN provided, saves log of error to prints/error_gmres/iN{iN}.txt

    Parameters:
    A : m x m complex (possibly) Non-Hermitian matrix (or function that applies it)
    b: m dim vector
    iN: Newton iteration number
    iA: Arclength iteration number
    N_gmres: maximum number of iterations for the algorithm (m being the largest possible)
    tol_gmres: threshold value for convergence

    Returns:
    x: m dim vector that minimizes ||Ax-b|| that lies in the Krylov subspace of dimension n (<m)
    e: error of each iteration, where the error is calculated as ||r_k||/||b|| where r_k is (b-A*x_k), the residue at 
    the k iteration.
    """

    b_norm = np.linalg.norm(b)
    e = [1.] #r_norm/b_norm
    n = N_gmres

    # Determine the data type based on b
    dtype = b.dtype

    #Initialize sine and cosine Givens 1d vectors. This allows the algorithm to be O(k) instead of O(k^2)
    sn = np.zeros(n,dtype=dtype)
    cs = np.zeros(n,dtype=dtype)

    #Unitary base of Krylov subspace (maximum number of cols: n)
    Q = np.zeros((len(b), n),dtype=dtype)
    Q[:,0] = b/b_norm #Normalize the input vector

    #Hessenberg matrix
    H = np.zeros((n+1,n),dtype=dtype)

    #First canonical vector:
    e1 = np.zeros(n+1,dtype = dtype)
    e1[0] = 1.

    #Beta vector to be multiplied by Givens matrices.
    beta = e1 * b_norm

    #In each iteration a new column of Q and H is computed.
    #The H column is then modified using Givens matrices so that H becomes a triangular matrix R
    for k in range(1,n):
        Q[:,k], H[:k+1,k-1] = arnoldi_step(A, Q, k) #Perform Arnoldi iteration to add column to Q (m entries) and to H (k entries)  
        
        H[:k+1,k-1], cs[k-1],sn[k-1] = apply_givens_rotation(H[:k+1,k-1],cs,sn,k) #eliminate the last element in H ith row and update the rotation matrix

        #update residual vector
        beta[k] = -sn[k-1].conj() * beta[k-1]
        beta[k-1] = cs[k-1] * beta[k-1]
        
        #||r_k|| can be obtained with the last element of beta because of the Givens algorithm
        error = abs(beta[k])/b_norm

        #save the error
        e.append(error)
        if gmres_dir:
            suffix = f'{iA:02}' if iA is not None else ''
            path = os.path.join(gmres_dir+suffix, f'gmres_iN{iN:02}.txt')
            print(f'{k},{error}', file=open(path, 'a'))

        if error<tol_gmres:
            break

    return H[:k,:k], beta[:k], Q[:,:k]

def solve_GMRES(H, beta, Q):
    """Solves the GMRES system given H, beta and Q matrices returned by GMRES function."""
    #Determine the data type based on beta
    dtype = beta.dtype
    #Solve for y in H y = beta
    y = backsub(H, beta)
    #Reconstruct x from y and Q
    x = Q @ y
    return x

def arnoldi_step(A, Q, k):
    """Performs k_th Arnoldi iteration of Krylov subspace spanned by <r, Ar, A^2 r,.., A^(k-1) r>"""
    # Check if A is a function or a matrix
    if callable(A):
        v = A(Q[:, k-1])  # generate candidate vector using the function
    else:
        v = A @ Q[:, k-1]  # generate candidate vector using matrix multiplication

    # Determine the data type based on Q and v
    dtype = v.dtype

    h = np.zeros(k+1, dtype=dtype)

    for j in range(k):#substract projections of previous vectors
        h[j] = np.dot(Q[:,j].conj(), v)
        v -= h[j] * Q[:,j]

    h[k] = np.linalg.norm(v, 2)
    return (v/h[k], h) #Returns k_th column of Q (python indexing) and k-1 column of H

def apply_givens_rotation(h, cs, sn, k):
    #Premultiply the last H column by the previous k-1 Givens matrices
    for i in range(k-1):
        temp = cs[i]*h[i] + sn[i]*h[i+1]
        h[i+1] = -sn[i].conj()*h[i] + cs[i].conj()*h[i+1]
        h[i] = temp        

    hip = np.sqrt(np.abs(h[k-1])**2+np.abs(h[k])**2)
    cs_k, sn_k = h[k-1].conj()/hip, h[k].conj()/hip

    #Update the last H entries and eliminate H[k,k-1] to obtain a triangular matrix R
    h[k-1] = cs_k*h[k-1] + sn_k*h[k]
    h[k] = 0
    return h, cs_k, sn_k


def backsub(R, b):
    """
    Solves the equation Rx = b, where R is a square right triangular matrix
    """
    n = len(b)

    # Determine the data type based on b
    dtype = b.dtype

    x = np.zeros(n,dtype=dtype)
    
    for i in range(n-1, -1, -1):
        aux = 0
        for j in range(i+1, n):
            aux += x[j] * R[i,j]
        x[i] = (b[i]-aux)/R[i,i]
    return x
    
def arnoldi_eig(A, b, n, tol):
    """
    Obtains approximate eigenvalues from A using Arnoldi iteration
    """

    # Determine the data type based on b
    dtype = b.dtype

    #Unitary base of Krylov subspace (maximum number of cols: n)
    Q = np.zeros((len(b), n+1), dtype=dtype)
    Q[:,0] = b/np.linalg.norm(b) #Normalize the input vector

    #Hessenberg matrix
    H = np.zeros((n+1,n), dtype=dtype)

    #In each iteration a new column of Q and H is computed.
    for k in range(1,n+1):
        Q[:,k], H[:k+1,k-1] = arnoldi_step(A, Q, k) #Perform Arnoldi iteration to add column to Q (m entries) and to H (k entries)  

        if H[k,k-1] < tol:

            break

    #Get H eigenvalues and eigvec
    eigval, eigvec = np.linalg.eig(H[:k,:k])

    return eigval, eigvec, Q[:,:k]

def test_gmres():
    m = 1000
    np.random.seed(0)

    A = np.random.randn(m,m)*m**(-.5) 
    b = np.random.randn(m)

    n = m//2
    n = 50
    tol = 1e-20

    # record start time
    time_start = time.perf_counter()
    # call benchmark code

    H, beta, Q = GMRES(A, b, n, tol)
    x = solve_GMRES(H, beta, Q)

    # record end time
    time_end = time.perf_counter()
    print('Performance GMRES:', time_end-time_start)

    print('GMRES rel. residual:', np.linalg.norm(b - A @ x)/np.linalg.norm(b))

    # record start time
    time_start = time.perf_counter()

    # call benchmark code
    x_scipy = scipy.linalg.solve(A, b)

    # call benchmark code
    # x_scipy,info = scipy.sparse.linalg.gmres(A, b, atol = tol, maxiter = 5, restart = 10, \
    #                                          callback = lambda x:print(x), callback_type = 'pr_norm')
    
    # record end time
    time_end = time.perf_counter()
    print('Performance Scipy:', time_end-time_start)
    print('Norm (scipy -  GMRES) = ', np.linalg.norm(x-x_scipy))
    print('Scipy convergenge:', info, 'rel. residual:', np.linalg.norm(b - A @ x_scipy)/np.linalg.norm(b))

def test_arnoldi():
    m = 1000
    np.random.seed(0)

    A = np.random.randn(m,m)*m**(-.5)

    A = 2*np.eye(m) + 0.5*A/np.sqrt(m)
    A = np.random.rand(m,m) + 0.5j*np.random.rand(m,m)

    b = np.random.randn(m)

    n = m//2
    tol = 1e-10

    eigval_H, eigvec_H, Q = arnoldi_eig(A, b, n, tol)

    print('Arnoldi iterations:', len(eigval_H))    

    eigval_A, eigvec_A = np.linalg.eig(A)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(eigval_A.real, eigval_A.imag, lw = 0, marker = 'o', s = 16)
    plt.scatter(eigval_H.real, eigval_H.imag, lw = 0, marker = 'o', s = 8)
    plt.xlabel('Re(eigenvalues)')
    plt.ylabel('Im(eigenvalues)')
    plt.show()


if __name__ == '__main__':
    test_gmres()
    # test_arnoldi()
