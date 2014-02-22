from NMF_algs import *
import matplotlib
import matplotlib.pyplot as plt

if 1:
    path1 = 'data/synthetic_200M_200_20/NMF_200M_200_20-qrr.txt'
    cols_path = 'data/synthetic_200M_200_20/NMF_200M_200_20-colnorms.txt'
    data1 = parse_normalized(path1, cols_path)
    data2 = parse(path1)
    path3 = 'data/synthetic_200M_200_20//NMF_200M_200_20-proj.txt'
    data3 = parse_normalized(path3, cols_path)
    
    orig_cols = [0, 1] + range(20, 200, 10)
    H_orig = []
    with open ('data/synthetic_200M_200_20/Hprime_20_200.txt') as f:
        for line in f:
            H_orig.append([float(v) for v in line.split()])
    H_orig = np.array(H_orig)
    H_orig_all = np.zeros((200, 200))
    for col in orig_cols:
        H_orig_all[col, col] = 1.0
    print H_orig.shape

    rs = []
    numcols = range(2, 22, 1)
    for cols in numcols:
        cols1, H1, resid1 = compute_extreme_pts(data1, cols, 'SPA', cols_path)
        cols2, H2, resid2 = compute_extreme_pts(data2, cols, 'xray')
        cols3, H3, resid3 = compute_extreme_pts(data3, cols, 'GP', cols_path)
        rs.append((resid1, resid2, resid3))
    visualize_resids(numcols, rs, 'synth_exact_residuals')
    '''

    r = 20
    # This is one of the biggest hacks I have ever done.  For some reason,
    # pyplot screws up the first plot.  Thus, we plot a dummy plot.
    cols0, H0, resid0 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    visualize(H0, cols0, 'dummy', 'synth_noisy_coeffs_dummy')
    cols0.sort()
    print cols0


    print H_orig
    not_orig_cols = [x for x in range(200) if x not in orig_cols]
    for ind1, i in enumerate(orig_cols):
        for ind2, j in enumerate(not_orig_cols):
            H_orig_all[i, j] = H_orig[ind1, ind2]
    imshow_wrapper(H_orig_all, title='Generation', fname='synth_exact_coeffs_gen')

    cols1, H1, resid1 = compute_extreme_pts(data1, r, 'SPA', cols_path)
    visualize(H1, cols1, 'SPA', 'synth_exact_coeffs_SPA')
    cols1.sort()
    print cols1

    cols2, H2, resid2 = compute_extreme_pts(data2, r, 'xray')
    visualize(H2, cols2, 'XRAY', 'synth_exact_coeffs_XRAY')
    cols2.sort()
    print cols2

    cols3, H3, resid3 = compute_extreme_pts(data3, r, 'GP', cols_path)
    visualize(H3, cols3, 'GP', 'synth_exact_coeffs_GP')
    cols3.sort()
    print cols3

    visualize_cols([cols1, cols2, cols3, orig_cols], H3.shape[1],
                   ['SPA', 'XRAY', 'GP', 'Generation'],
                   'synth_exact_cols')

    '''


