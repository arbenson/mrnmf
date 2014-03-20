#!/usr/bin/env dumbo

"""
Generate large synthetic (near-)separable tall-and-skinny matrices.

Usage:

     dumbo start GenSyntheticSepLarge.py \
     -hadoop $HADOOP_INSTALL \
	 -m [number of rows in millions] \
     -H [path to r x n coefficient matrix H] \
     -mat [HDFS path to matrix with one million records keys] \
     -epsilon [noise level] \
     -output [name of output file]

Example usage:

     dumbo start GenSyntheticSepLarge.py -hadoop $HADOOP_INSTALL \
     -m 200 -H 'data/Hprime_20_200.txt' -epsilone 1e-3 \
     -mat Simple_1M.txt -output Noisy_200M_200_20.bseq
"""

import sys
import os
import time
import random

import numpy as np
import struct

import util
import mrnmf

import dumbo
import dumbo.backends.common

# create the global options structure
gopts = util.GlobalOptions()

class Map:
  def __init__(self, m, mat_name, epsilon):
	  self.epsilon = epsilon
	  self.Hprime = []
	  try:
		  f = open(mat_name)
	  except:
		  f = open(mat_name.split('/')[-1])
	  for line in f:
		  row = [float(v) for v in line.split()]
		  self.Hprime.append(row)
	  f.close()

	  # matrix dimensions (m in millions)
	  self.m = m
	  self.n = len(self.Hprime[0])
	  self.r = len(self.Hprime)

	  self.Hprime = np.array(self.Hprime)

  def __call__(self, key, value):
	  W = np.random.random((self.m, self.r))
	  M = np.dot(W, np.hstack((np.eye(self.r), self.Hprime)))
	  # permutation of columns
	  P = range(self.n)
	  for i in xrange(self.r):
		  P[i], P[i * (n / r)] = P[i * (n / r)], P[i]
	  M = M[:, P]

	  # Add noise
	  N = np.random.random((self.m, self.n)) * self.epsilon
	  M += N

	  for row in M:
		  key = [np.random.random() for i in xrange(3)]
		  val = [float(v) for v in row]
		  yield key, struct.pack('d' * len(val), *val)

def runner(job):
	m = gopts.getintkey('m')
	epsilon = gopts.getfloatkey('epsilon')
	H = gopts.getstrkey('H')
    options=[('numreducetasks', '0'), ('nummaptasks', '40')]
    job.additer(mapper=Map(m, H, epsilon), reducer=mrnmf.ID_REDUCER, opts=options)

def starter(prog):
    print "running starter!"

    mypath =  os.path.dirname(__file__)
    print "my path: " + mypath

    # set the global opts
    gopts.prog = prog

    mat = prog.delopt('mat')
    if not mat:
        return "'mat' not specified"

    prog.addopt('memlimit','8g')

    gopts.getstrkey('reduce_schedule', '1')

	gopts.getintkey('m', 200)
	gopts.getfloatkey('epsilon', 0.0)

	H = prog.delopt('H')
	if not H:
		return "'H' not specified"
	gopts.getstrkey('H')

    prog.addopt('file',os.path.join(mypath, 'util.py'))
    prog.addopt('file',os.path.join(mypath, 'mrnmf.py'))
    prog.addopt('file',os.path.join(mypath, H))

    prog.addopt('input', mat)
    matname,matext = os.path.splitext(mat)

    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-sep%s'%(matname,'.bseq'))

    prog.addopt('overwrite','yes')
    prog.addopt('jobconf','mapred.output.compress=true')

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
