import sys
import numpy as np
from scipy import optimize

"""
After the MapReduce codes have generated data to reduce the dimension of the
problem, this script can be used to run standard NMF algorithms to compute the
extreme columns and the coefficient matrix H.
"""

def SPA(X, r):
    """ Successive projection algorithm (SPA) for NMF.  This algorithm computes
    the column indices.

    Args:
        X: The data matrix.
        r: The target separation rank.

    Returns:
        A list of r columns chosen by SPA.
    """
    cols = []
    m, n = X.shape
    assert(m == n)
    for _ in xrange(r):
        col_norms = np.sum(np.abs(X) ** 2, axis=0)
        col_ind = np.argmax(col_norms)
        cols.append(col_ind)
        col = np.reshape(X[:, col_ind], (n, 1))
        X = np.dot((np.eye(n) - np.dot(col, col.T) / col_norms[col_ind]), X)

    return cols

def col2norm(X):
    """ Compute all column 2-norms of a matrix. """
    return np.sum(np.abs(X) ** 2,axis=0)

def xray(X, r):
    """ X-ray algorithm for NMF.  This algorithm computes the column indices.

    Args:
        X: The data matrix.
        r: The target separation rank.

    Returns:
        A list of r columns chosen by X-ray.
    """
    cols = []
    R = np.copy(X)
    while len(cols) < r:
        i = np.argmax(col2norm(X))
        # Loop until we choose a column that has not been selected.
        while True:
            p = np.random.random((X.shape[0], 1))
            scores = col2norm(np.dot(R.T, X)) / col2norm(X)
            scores[cols] = -1   # IMPORTANT
            best_col = np.argmax(scores)
            if best_col in cols:
                # Re-try
                continue
            else:
                cols.append(best_col)
                H, rel_res = NNLSFrob(X, cols)
                R = X - np.dot(X[:, cols] , H)
                break
    return cols

def GP_cols(data, r):
    """ X-ray algorithm for NMF.  This algorithm computes the column indices.

    Args:
        data: The matrix G * X, where X is the nonnegative data matrix and G is
            a matrix with Gaussian i.i.d. random entries.
        r: The target separation rank.

    Returns:
        A list of r columns chosen by Gaussian projection.
    """
    votes = {}
    for row in data:
        min_ind = np.argmin(row)
        max_ind = np.argmax(row)
        for ind in [min_ind, max_ind]:
            if ind not in votes:
                votes[ind] = 1
            else:
                votes[ind] += 1

    votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in votes][0:r]

def NNLSFrob(X, cols):
    """ Compute H, the coefficient matrix, by nonnegative least squares to minimize
    the Frobenius norm.  Given the data matrix X and the columns cols, H is

             \arg\min_{Y \ge 0} \| X - X(:, cols) H \|_F.

    Args:
        X: The data matrix.
        cols: The column indices.

    Returns:
        The matrix H and the relative resiual.
    """
    ncols = X.shape[1]
    H = np.zeros((len(cols), ncols))
    for i in xrange(ncols):
        sol, res = optimize.nnls(X[:, cols], X[:, i])
        H[:, i] = sol
    rel_res = np.linalg.norm(X - np.dot(X[:, cols], H), 'fro')
    rel_res /= np.linalg.norm(X, 'fro')
    return H, rel_res

def ComputeNMF(data, r, alg, colnorms=None):
    """ Compute an approximate separable NMF of the matrix data.  By compute,
    we mean choose r columns and a best fitting coefficient matrix H.  The
    r columns are selected by the 'alg' option, which is one of 'SPA', 'xray',
    or 'GP'.  The coefficient matrix H is the one that produces the smallest
    Frobenius norm error.
    
    Args:
        data: The data matrix.
        r: The target separation rank.
        alg: Choice of algorithm for computing the columns.  One of 'SPA',
            'xray', or 'GP'.
        colnorms: If provided, the column norms.  Default is None, i.e.,
            column norms are not needed.

    Returns:
        The selected columns, the matrix H, and the relative residual.
    """
    data = np.copy(data)
    _, S, Vt = np.linalg.svd(data)
    A = np.dot(np.diag(S), Vt)
    A = np.array(data)
    if alg == 'SPA':
        cols = SPA(A, r)
    elif alg == 'xray':
        cols = xray(A, r)
    elif alg == 'GP':
        cols = GP_cols(data, r)
    else:
        raise Exception('Unknown algorithm: %s' % str(alg))
        
    if colnorms != None:
        A = np.dot(A, np.diag(colnorms))

    H, rel_res = NNLSFrob(A, cols)
    return cols, H, rel_res

def ParseColnorms(colpath):
	norms = []
	with open(colpath, 'r') as f:
		for line in f:
			norms.append(float(line.split()[-1]))
	return norms

def ParseMatrix(matpath):
	matrix = []
	with open(matpath, 'r') as f:
		for line in f:
			matrix.append([float(v) for v in row.split[1:]])
	return np.array(matrix)

if __name__ == "__main__":
    data = sys.argv[1]
    alg = sys.argv[2]
	if len(argv) > 3:
		cols = sys.argv[3]
    cols = sys.argv[2]print sys.argv 

