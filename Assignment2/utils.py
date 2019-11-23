import numpy as np
import scipy.sparse as sparse
import scipy as sp
from scipy.sparse.linalg import spsolve, lsmr
import matplotlib.pyplot as plt

def normalizationMatrix(x):
    """
    Returns the transformation matrix used to normalize the inputs x.
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the centre.
    Input:     x    2xN
    Output:    T    3x3     transformation matrix of points
    """
    x2d = x[:2, :]
    n = x.shape[1]
    
    # TO DO TASK:
    #--------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    #--------------------------------------------------------------
    
    # Get centroid and mean-distance to centroid
    centroid = np.mean(x2d, 2)
    meanDist = np.mean(np.sqrt(np.sum(x2d - np.title(centroid, np.block(1,n))) ** 2))
    T = np.block([[np.sqrt(2) / meanDist, 0, np.dot(- centroid(1), np.sqrt(2)) / meanDist], [0, np.sqrt(2) / meanDist, np.dot(- centroid(2), np.sqrt(2)) / meanDist], [0, 0, 1]])
    
    return T

# def eightPointsAlgorithm(x1,x2):
#     """ Computes the fundamental matrix from corresponding points
#     (x1,x2 3*n arrays) using the normalized 8 point algorithm.
#     each row is constructed as
#     [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1] """
#     n = x1.shape[1]
#     if x2.shape[1] != n:
#         raise ValueError("Number of points don’t match.")
#     # build matrix for equations
#     A = np.zeros((n,9))
#     for i in range(n):
#         A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
#                 x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
#                 x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
#     # compute linear least square solution
#     U,S,V = np.linalg.svd(A)
#     F = V[-1].reshape(3,3)
#     # constrain F
#     # make rank 2 by zeroing out last singular value
#     U,S,V = np.linalg.svd(F)
#     S[2] = 0
#     F = np.dot(U, np.dot(np.diag(S),V))

#     return F

# def eightPointsAlgorithm(p1, p2):
#     """
#     Calculates the fundamental matrix between two views using the normalized 8 point algorithm
#     Inputs: 
#                     x1      3xN     homogeneous coordinates of matched points in view 1
#                     x2      3xN     homogeneous coordinates of matched points in view 2
#     Outputs:
#                     F       3x3     fundamental matrix
#     """
#     N = p1.shape[1]

#     # Construct transformation matrices to normalize the coordinates
#     p1 = np.copy(p1)
#     n1 = p1.shape[1]
#     p1 = np.resize(p1, (3, n1))
#     p1[2,:] = 1
#     p1 = np.matrix(p1)
    
#     p2 = np.copy(p2)
#     n2 = p2.shape[1]
#     p2 = np.resize(p2, (3, n2))
#     p2[2,:] = 1
#     p2 = np.matrix(p2)
    
#     # Normalize inputs
#     m = np.mean(p1,1)
#     d = np.mean(np.sqrt(np.sum(np.power(p1,2),1))) # mean distance
#     s = np.sqrt(2) # want points to have mean distance sqrt(2)
#     T1 = np.matrix(np.diag([s/d,s/d,1]))*np.matrix([[1,0,-m[0]],[0,1,-m[1]],[0,0,1]])
#     p1 = T1*p1 # apply transformation
    
#     m = np.mean(p2,1)
#     d = np.mean(np.sqrt(np.sum(np.power(p2,2),1))) # mean distance
#     s = np.sqrt(2) # want points to have mean distance sqrt(2)
#     T2 = np.matrix(np.diag([s/d,s/d,1]))*np.matrix([[1,0,-m[0]],[0,1,-m[1]],[0,0,1]])
#     p2 = T2*p2 # apply transformation
    
#     # Construct matrix A encoding the constraints on x1 and x2
#     x1, y1 = np.array(p1[0,:].T), np.array(p1[1,:].T)
#     x2, y2 = np.array(p2[0,:].T), np.array(p2[1,:].T)
#     A = np.hstack((x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones((N,1))))

#     U,s,Vt = sp.linalg.svd(np.matrix(A))
#     F = Vt[-1,:].T.reshape((3,3))

#     # Solve for f using SVD
#     U,s,Vt = sp.linalg.svd(F)
    
#     # Enforce that rank(F)=2
#     F = np.matrix(U)*np.matrix(np.diag([s[0],s[1],0]))*np.matrix(Vt)
    
#     # Transform F back
#     F = np.dot(U, np.dot(np.diag(s),Vt))
    
#     return F

def eightPointsAlgorithm(x1, x2):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs: 
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    
    # Construct matrix A encoding the constraints on x1 and x2
    A = np.zeros((N,9))
    for i in range(N):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # Solve for f using SVD
    U,S,V = np.linalg.svd(F)
    
    # Enforce that rank(F)=2
    S[2] = 0
    
    # Transform F back
    F = np.dot(U, np.dot(np.diag(S),V))
    
    return F


def right_epipole(F):
    """ 
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.) 
    """
    
    # The epipole is the null space of F (F * e = 0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    
    return e


def plot_epipolar_line(im, F, x, e):
    """ 
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    
    plt.plot(x[0], x[1], 'r*')

    m, n = im.shape[:2]
    line = np.dot(F, x)
    # epipolar line parameter and values

    t = np.linspace(0, n, 100)

    lt = np.array(
        [(line[2]+line[0]*tt)/(-line[1]) for tt in t]
    )
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)
    if e is None:
        e = right_epipole(F)
    plt.plot(e[0]/e[2], e[1]/e[2], 'b*')
    

def decomposeE(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0,0,0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # TODO: Compute possible rotations and translations
    U,S,V = np.linalg.svd(E)
    
    if (abs(S[1,1] - S[2,2]) > 0.01 or S[3,3] != 0):
        S = np.diag(np.concatenate([1,1,0]))
    
    W = np.concatenate([[0, - 1, 0], [1, 0, 0], [0, 0, 1]])
    T = np.dot(np.dot(np.dot(U, S), W), U.T)
    R1 = np.dot(np.dot(U, W.T), V.T)
    R2 = np.dot(np.dot(U, W), V.T)
    t1 = np.concatenate([[T(3, 2)], [T(1, 3)], [T(2, 1)]])
    t2 = -t1

    # End of your code
    
    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer3D(x1[:,0:1], x2[:,0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0,2], test[1,2], test[2,2], test[3,2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]
    
    return Pl, Pr

def infer3D(x1, x2, Pl, Pr):
    """
    Infers 3D positions of the point-correspondences x1 and x2, using 
    the rotation matrices Rl, Rr and translation vectors tl, tr. 
    Using a least-squares approach.
    """

    M = x1.shape[1]
    
    # Extract rotation and translation
    Rl = Pl[:3,:3]
    tl = Pl[:3,3]
    Rr = Pr[:3,:3]
    tr = Pr[:3,3]

    # Construct matrix A with constraints on 3d points
    rowIdx = np.tile(np.arange(4*M), (3, 1)).T.reshape(-1)
    colIdx = np.tile(np.arange(3*M), (1, 4)).reshape(-1)
    
    A = np.zeros((4*M, 3))
    A[:M, :3] = x1[0:1,:].T @ Rl[2:3,:] - np.tile(Rl[0:1,:], (M, 1))
    A[M:2*M, :3] = x1[1:2,:].T @ Rl[2:3,:] - np.tile(Rl[1:2,:], (M, 1))
    A[2*M:3*M, :3] = x2[0:1,:].T @ Rr[2:3,:] - np.tile(Rr[0:1,:], (M, 1))
    A[3*M:4*M, :3] = x2[1:2,:].T @ Rr[2:3,:] - np.tile(Rr[1:2,:], (M, 1))
    
    A = sparse.csr_matrix((A.reshape(-1), (rowIdx, colIdx)), shape=(4*M, 3*M))
    
    # Construct vector b
    b = np.zeros((4*M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1,:].T * tl[2]
    b[M:2*M] = np.tile(tl[1], (M, 1)) - x1[1:2,:].T * tl[2]
    b[2*M:3*M] = np.tile(tr[0], (M, 1)) - x2[0:1,:].T * tr[2]
    b[3*M:4*M] = np.tile(tr[1], (M, 1)) - x2[1:2,:].T * tr[2]
    
    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M,3).T
    
    return x3d