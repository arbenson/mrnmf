"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich, 
   Purdue University, and Stanford University.
   All rights reserved.
 
   This file is part of MRNMF and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
"""

import numpy as np
import sys
from scipy import optimize
from cvxopt import matrix, solvers

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

def visualize_resids(numcols, rs, fname=None):
    fig = plt.figure()
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}
    matplotlib.rc('font', **font)
    #rcParams.update({'figure.autolayout': True})
    
    markers = ['b-*', 'g-o', 'r-<']
    for j in xrange(len(rs[0])):
        plt.plot(numcols, [x[j] for x in rs], markers[j])
    plt.legend(['SPA', 'XRAY', 'GP'])
    plt.xlabel('Separation rank (r)')
    plt.ylabel('Relative error')
    F = plt.gcf()
    F.subplots_adjust(bottom=0.15)
    F.subplots_adjust(left=0.20)
    plt.show()
    F.set_size_inches((4, 4))
    if fname != None:
        fig.savefig(fname + '.eps')

def imshow_wrapper(H, title=None, fname=None, size=(2.2, 2.2), adjust=0.):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    rcParams.update({'figure.autolayout': True})
    
    plt.imshow(H, cmap=cm.Greys)
    plt.colorbar()
    plt.xlabel('column index')
    plt.ylabel('row index')
    if title == None:
        plt.title('Entries of H')
    else:
        plt.title(title)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    F = plt.gcf()
    F.subplots_adjust(left=adjust)
    plt.show()
    F.set_size_inches(size)
    if fname != None:
        fig.savefig(fname + '.eps')


def visualize(Hprime, cols, title=None, fname=None):
    n = Hprime.shape[1]
    H = np.zeros((n, n))
    H[cols, :] = Hprime
    imshow_wrapper(H, title, fname)
    

def visualize_cols(all_cols, n, legend, fname=None):
    fig = plt.figure()
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 6}
    matplotlib.rc('font', **font)
    rcParams.update({'figure.autolayout': True})

    markers = ['*', 'o', '<', '.']
    for i, cols in enumerate(all_cols):
        cols = [c + 1 for c in cols]
        plt.plot(cols, [1 - 0.1 * i] * len(cols), markers[i])
    plt.xlabel('column index')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1 - 2, x2 + 0.5, 1 - 0.1 * (len(all_cols) + 3), 1.05))
    plt.legend(legend, loc=4)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.title('Selected columns')
    F = plt.gcf()
    #F.subplots_adjust(bottom=0.15)
    plt.show()
    F.set_size_inches((2, 2))
    if fname != None:
        fig.savefig(fname + '.eps')
        
def parse(path):
    data = []
    try:
        f = open(path, 'r')
    except:
        # We may be expecting only the file to be distributed with the script
        f = open(path.split('/')[-1], 'r')
    mat = f.read()
    f.close()
    for line in mat.split('\n')[:-1]:
        row = [float(v.rstrip(']')) \
                   for v in line[line.rfind('[') + 1:line.rfind(']')].split(',')]
        data.append(row)
    return np.array(data)

def parse_normalized(path, colnorms_path, unnormalize=False):
    data = parse(path)
    with open(colnorms_path, 'r') as f:
        norms = []
        for line in f:
            norms.append(float(line.split()[-1]))
    mult = np.mat(np.linalg.inv(np.diag(norms)))
    if unnormalize:
        mult = np.mat(np.diag(norms))
    return np.dot(data, mult)

def HottTopixx(M, epsilon, r):
    # Treating variables in X row-major
    n = M.shape[1]
    p = np.random.random((n, 1))
    c = matrix(np.kron(p, np.eye(n, 1)))
    
    # tr(X) = r
    A = matrix(np.kron(np.ones((1, n)), np.array(([1] + [0] * (n-1)))))
    b = matrix([float(r)]) # need float cast
    
    # X(i, i) \le 1 for all i
    G1 = np.zeros((n, n * n))
    for i in xrange(n):
        G1[i, n * i] = 1
    h1 = np.ones((n, 1))
    
    # X(i, j) \le X(i, i) for all i, j
    G2 = np.kron(np.eye(n), np.hstack((-np.ones((n-1, 1)), np.eye(n-1))))
    h2 = np.zeros(((n-1) * n, 1))
    
    # X(i, j) \ge 0 for all i, j
    G3 = -np.eye(n * n)
    h3 = np.zeros((n * n, 1))
    
    # \| M - MX \|_1 \le 2\epsilon
    # We are not going to assume that M is nonnegative, so we
    # turn the one norm constraint into two sets of constraints.
    m = M.shape[0]
    G4 = np.kron(-M, np.ones((1, n)))
    h4 = np.reshape(-np.sum(M, axis=1) + 2 * epsilon, (m, 1))
    G5 = np.kron(M, np.ones((1, n)))
    h5 = np.reshape(np.sum(M, axis=1) + 2 * epsilon, (m, 1))
    
    # min c^Ty
    # s.t. Gy + s = h
    #      Ay = b
    #      s \ge 0
    G = matrix(np.vstack((G1, G2, G3, G4, G5)))
    h = matrix(np.vstack((h1, h2, h3, h4, h5)))
    X = np.reshape(np.array(solvers.lp(c, G, h, A=A, b=b)['x']), (n, n))
    return list(np.argsort(np.diag(X))[-r:])

def SPA(A, r):
    cols = []
    m, n = A.shape
    assert(m == n)
    for _ in xrange(r):
        col_norms = np.sum(np.abs(A) ** 2, axis=0)
        col_ind = np.argmax(col_norms)
        cols.append(col_ind)
        col = np.reshape(A[:, col_ind], (n, 1))
        A = np.dot((np.eye(n) - np.dot(col, col.T) / col_norms[col_ind]), A)

    return cols

def col2norm(A):
	return np.sum(np.abs(A) ** 2,axis=0)

def xray(X, r):
	cols = []
	R = np.copy(X)
	while len(cols) < r:
            i = np.argmax(col2norm(X))
            while True:
                i = np.random.choice(range(X.shape[1]))
                if i not in cols:
                    break
            Ri = R[:, i]
            p = np.random.random((X.shape[0], 1))
            scores = col2norm(np.dot(R.T, X)) / col2norm(X)
            scores[cols] = -1   # IMPORTANT
            best_col = np.argmax(scores)
            if best_col in cols:
                # Re-try
                continue
            if best_col not in cols:
                cols.append(best_col)
            H, rel_res = NNLSFrob(X, cols)
            R = X - np.dot(X[:, cols] , H)
        return cols

def GP_cols(data, r):
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

def NNLSFrob(A, cols):
    ncols = A.shape[1]
    H = np.zeros((len(cols), ncols))
    for i in xrange(ncols):
        sol, res = optimize.nnls(A[:, cols], A[:, i])
        H[:, i] = sol
    rel_res = np.linalg.norm(A - np.dot(A[:, cols], H), 'fro')
    rel_res /= np.linalg.norm(A, 'fro')
    return H, rel_res

def compute_extreme_pts(data, r, alg, colpath=None):
    data = np.copy(data)
    _, S, Vt = np.linalg.svd(data)
    A = np.dot(np.diag(S), Vt)
    A = np.array(data)
    if alg == 'SPA':
        cols = SPA(np.copy(A), r)
    elif alg == 'xray':
        cols = xray(np.copy(A), r)
    elif alg == 'GP':
        cols = GP_cols(data, r)
    elif alg == 'Hott':
        epsilon = 1e-5
        cols = HottTopixx(A, epsilon, r)
    else:
        raise Exception('Unknown algorithm: %s' % str(alg))
        
    if colpath != None:
        with open(colpath, 'r') as f:
            norms = []
            for line in f:
                norms.append(float(line.split()[-1]))
        A = np.dot(A, np.diag(norms))

    H, rel_res = NNLSFrob(A, cols)
    return cols, H, rel_res
