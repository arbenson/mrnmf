import numpy as np

"""
Generate a small, noisy r-separable matrix of size 10,0000 x 10,
with separation rank r = 4.
"""

m = 10000
n = 10
r = 4
epsilon = 1e-3

Hprime = np.random.random((r, n - r))
W = np.random.random((m, r))
M = np.dot(W, np.hstack((np.eye(r), Hprime)))

# Add noise
N = np.random.random((m, n)) * epsilon
M += N

with open('SmallNoisySep_10k_10_4.txt', 'w') as f:
	for row in M:
		row = [str(v) for v in row]
		f.write(' '.join(row) + '\n')
