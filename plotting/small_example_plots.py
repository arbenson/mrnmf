from NMF_algs import *
import matplotlib
import matplotlib.pyplot as plt

# Locations of data
path1 = '../small.out-qrr.txt'
cols_path = '../small.out-colnorms.txt'
data1 = parse_normalized(path1, cols_path)
data2 = parse(path1)
path3 = '../small.out-proj.txt'
data3 = parse_normalized(path3, cols_path)
    
# Test for residuals.
rs = []
numcols = range(2, 22, 1)
for cols in numcols:
	cols1, H1, resid1 = compute_extreme_pts(data1, cols, 'SPA', cols_path)
	cols2, H2, resid2 = compute_extreme_pts(data2, cols, 'xray')
	cols3, H3, resid3 = compute_extreme_pts(data3, cols, 'GP', cols_path)
	rs.append((resid1, resid2, resid3))
visualize_resids(numcols, rs, 'synth_exact_residuals')

# Visualize columns when r = 4.
r = 4
cols1, H1, resid1 = compute_extreme_pts(data1, r, 'SPA', cols_path)
visualize(H1, cols1, 'SPA', 'synth_exact_coeffs_SPA')
cols1.sort()

cols2, H2, resid2 = compute_extreme_pts(data2, r, 'xray')
visualize(H2, cols2, 'XRAY', 'synth_exact_coeffs_XRAY')
cols2.sort()

cols3, H3, resid3 = compute_extreme_pts(data3, r, 'GP', cols_path)
visualize(H3, cols3, 'GP', 'synth_exact_coeffs_GP')
cols3.sort()

visualize_cols([cols1, cols2, cols3, orig_cols], H3.shape[1],
			   ['SPA', 'XRAY', 'GP', 'Generation'],
			   'synth_exact_cols')

