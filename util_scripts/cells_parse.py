"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich, 
   Purdue University, and Stanford University.
   All rights reserved.
 
   This file is part of MRNMF and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
"""

import csv
import numpy as np

mat = []
with open('cells_example_1.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for i, row in enumerate(reader):
		if i == 0: continue
		mat.append([float(v) for v in row])

with open('cells_example_1.txt', 'w') as f:
	for row in mat:
		row = [str(v) for v in row]
		f.write(' '.join(row) + '\n')

with open('cells_example_1_30k.txt', 'w') as f:
	for i, row in enumerate(mat):
		if i >= 30000: continue
		row = [str(v) for v in row]
		f.write(' '.join(row) + '\n')
