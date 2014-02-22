from NMF_algs import *
import matplotlib
import matplotlib.pyplot as plt

if 0:
    '''
    Testing artificial matrices
    '''
    m = 100
    n = 100
    r = 10
    #epsilon = 1e-3
    epsilon = 0
    Hprime = np.random.random((r, n-r))
    for i in xrange(n-r):
        Hprime[:, i] /= np.linalg.norm(Hprime[:, i], 1)
    
    W = np.random.random((m, r))
    M = np.hstack((W, np.dot(W, Hprime)))
    # permutation of columns
    
    P1 = np.arange(0, n, n / r)
    P2 = [x for x in np.arange(0, n) if x not in P1]
    P = list(P1) + P2
    M = M[:, P]

    # Add noise
    N = np.random.random((m, n)) * epsilon
    M += N 
    _, R = np.linalg.qr(M)

    data = R
    cols, H, resid = compute_extreme_pts(data, r, 'xray')
    visualize(H, cols)
    print cols
    print resid

if 0:
    path = 'data/NMF_200M_200_20-qrr.txt'
    data = parse(path)
    cols, H, resid = compute_extreme_pts(data, 25, 'xray')
    visualize(H, cols)
    print cols
    print resid

if 0:
    path = 'data/NMF_200M_320_40_noisy-qrr.txt'
    data = parse(path)
    cols, H, resid = compute_extreme_pts(data, 40, 'xray')
    print cols
    print resid

if 0:
    path1 = 'data/cells-kron-40k-qrr.txt'
    data1 = parse(path1)
    path2 = 'data/cells-kron-40k-proj.txt'
    data2 = parse(path2)
    
    rs = []
    num_cols = range(2, 26, 2)
    for i in num_cols:
        cols, H, resid1 = compute_extreme_pts(data1, i, 'SPA')
        cols, H, resid2 = compute_extreme_pts(data1, i, 'xray')
        cols, H, resid3 = compute_extreme_pts(data2, i, 'GP')
        rs.append((resid1, resid2, resid3))

    plt.plot(num_cols, rs)
    plt.xlabel('nonnegative rank')
    plt.ylabel('relative residual')
    plt.title('A kron A first 40k rows of A')
    plt.legend(['SPA', 'XRAY', 'GP'])
    plt.show()

    r = 12
    cols1, H1, resid1 = compute_extreme_pts(data1, 10, 'SPA')
    visualize(H1, cols1, 'Coefficients: SPA')
    print cols1

    cols2, H2, resid2 = compute_extreme_pts(data1, 10, 'xray')
    visualize(H2, cols2, 'Coefficients: XRAY')
    print cols2

    cols3, H3, resid3 = compute_extreme_pts(data2, 10, 'GP')
    visualize(H3, cols3, 'Coefficients: GP')
    print cols3

    visualize_cols([cols1, cols2, cols3], H3.shape[1], ['SPA', 'XRAY', 'GP'])

if 0:
    path1 = 'data/cells-kron-40k-qrr.txt'


