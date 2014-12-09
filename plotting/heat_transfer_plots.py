"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich, 
   Purdue University, and Stanford University.
   All rights reserved.
 
   This file is part of MRNMF and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
"""

from NMF_algs import *
import matplotlib
import matplotlib.pyplot as plt

if 1:
    path1 = 'data/heat-transfer/heat-transfer-normalized-qrr.txt'
    data1 = parse(path1)
    cols_path = 'data/heat-transfer/heat-transfer-colnorms.txt'
    data2 = parse_normalized(path1, cols_path, unnormalize=True)
    path3 = 'data/heat-transfer/heat-transfer-normalized-proj.txt'
    data3 = parse_normalized(path3, cols_path)

    rs = []
    numcols = range(1, 64, 2)
    for cols in numcols:
        cols1, H1, resid1 = compute_extreme_pts(data1, cols, 'SPA', cols_path)
        cols2, H2, resid2 = compute_extreme_pts(data2, cols, 'xray')
        cols3, H3, resid3 = compute_extreme_pts(data3, cols, 'GP', cols_path)
        rs.append((resid1, resid2, resid3))
    visualize_resids(numcols, rs, 'heat_residuals')
    
    r = 10
    # This is one of the biggest hacks I have ever done.  For some reason,
    # pyplot screws up the first plot.  Thus, we plot a dummy plot.
    cols0, H0, resid0 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    visualize(H0, cols0, 'dummy', 'synth_noisy_coeffs_dummy')
    cols0.sort()
    print cols0

    cols1, H1, resid1 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    """
    visualize(H1, cols1, 'SPA',
              'heat_coeffs_SPA')
    """

    v1 = H1[cols1.index(0), 0:34]
    v2 = H1[cols1.index(33), 0:34]
    fig = plt.figure()
    inds = range(0, 34)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}
    matplotlib.rc('font', **font)
    rcParams.update({'figure.autolayout': True})
    plt.plot(inds, v1, 'b-*')
    plt.plot(inds, v2, 'g-o')
    plt.legend(['Col. 1', 'Col. 34'], loc=6)
    plt.xlabel('Column index')
    plt.ylabel('Coefficient matrix (H) value')
    F = plt.gcf()
    plt.show()
    F.set_size_inches((3, 3))
    fig.savefig('heat_combinations.eps')

    cols1.sort()
    print cols1

    cols2, H2, resid2 = compute_extreme_pts(data2, r, 'xray')
    visualize(H2, cols2, 'XRAY',
              'heat_coeffs_XRAY')
    cols2.sort()
    print cols2

    cols3, H3, resid3 = compute_extreme_pts(data3, r, 'GP', cols_path)
    visualize(H3, cols3, 'GP',
              'heat_coeffs_GP')
    cols3.sort()
    print cols3

    visualize_cols([cols1, cols2, cols3], H3.shape[1], ['SPA', 'XRAY', 'GP'],
                   'heat_cols')

