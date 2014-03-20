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
    path1 = 'data/synthetic_200M_200_20/NMF_200M_200_20_noisy-qrr.txt'
    cols_path = 'data/synthetic_200M_200_20/NMF_200M_200_20_noisy-colnorms.txt'
    data1 = parse_normalized(path1, cols_path)
    data2 = parse(path1)
    path3 = 'data/synthetic_200M_200_20//NMF_200M_200_20_noisy-proj.txt'
    data3 = parse_normalized(path3, cols_path)

    rs = []
    numcols = range(1, 22, 1)
    for cols in numcols:
        cols1, H1, resid1 = compute_extreme_pts(data1, cols, 'SPA', cols_path)
        cols2, H2, resid2 = compute_extreme_pts(data2, cols, 'xray')
        cols3, H3, resid3 = compute_extreme_pts(data3, cols, 'GP', cols_path)
        rs.append((resid1, resid2, resid3))
    visualize_resids(numcols, rs, 'synth_noisy_residuals')

    '''
    r = 20
    # This is one of the biggest hacks I have ever done.  For some reason,
    # pyplot screws up the first plot.  Thus, we plot a dummy plot.
    cols0, H0, resid0 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    visualize(H0, cols0, 'dummy', 'synth_noisy_coeffs_dummy')
    cols0.sort()
    print cols0

    cols1, H1, resid1 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    visualize(H1, cols1, 'SPA', 'synth_noisy_coeffs_SPA')
    cols1.sort()
    print cols1

    cols2, H2, resid2 = compute_extreme_pts(data2, r, 'xray')
    print H2[3, :]
    visualize(H2, cols2, 'XRAY', 'synth_noisy_coeffs_XRAY')
    cols2.sort()
    print cols2

    cols3, H3, resid3 = compute_extreme_pts(data3, r, 'GP', cols_path)
    visualize(H3, cols3, 'GP', 'synth_noisy_coeffs_GP')
    cols3.sort()
    print cols3

    orig_cols = [0, 1] + range(20, 200, 10)
    visualize_cols([cols1, cols2, cols3, orig_cols], H3.shape[1],
                   ['SPA', 'XRAY', 'GP', 'Generation'],
                   'synth_noisy_cols')
    '''
