#!/usr/bin/env python

from cvxopt import matrix, solvers
import numpy as np
import time

def HotTopixx(M, epsilon, r):
    # Treating variables in X row-major

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
    # By the above constratins, any row of M - MX is nonnegative. Thus, we
    # can turn the one norm constraint into a set of linear constraints
    G4 = np.kron(-M, np.ones((1, n)))
    h4 = np.reshape(-np.sum(M, axis=1) + 2 * epsilon, (m, 1))
    
    # min c^Ty
    # s.t. Gy + s = h
    #      Ay = b
    #      s \ge 0
    G = matrix(np.vstack((G1, G2, G3, G4)))
    h = matrix(np.vstack((h1, h2, h3, h4)))
    X = np.reshape(np.array(solvers.lp(c, G, h, A=A, b=b)['x']), (n, n))
    return list(np.argsort(np.diag(X))[-r:])

# setup
# NMF: M = WH, M is m x n, W is m x r, H is r x n
m = 10000
n = 32
r = 6

Hprime = np.random.random((r, n-r) )
W = np.random.random((m, r))
N = np.random.random((m, n)) * 1e-5
epsilon = 1.1 * np.linalg.norm(N, 1)
M = np.dot(W, np.hstack((np.eye(r), Hprime))) + N
# permutation of columns
P = np.random.choice(n, n, replace=False)
M = M[:, P]
M = np.dot(M, np.linalg.inv(np.diag(np.sum(M, axis=0)))) # normalize

t0 = time.time()
topics = HotTopixx(M, epsilon, r)
print 'The hot topixx are: ' + str(topics)
print 'Running time: ' + str(time.time() - t0)
