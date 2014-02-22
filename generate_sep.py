#!/usr/bin/env dumbo

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
  def __init__(self, mat_name):
	  self.Hprime = []
	  with open(mat_name) as f:
		  for line in f:
			  row = [float(v) for v in line.split()]
			  self.Hprime.append(row)
	  self.Hprime = np.array(self.Hprime)

  def __call__(self, key, value):
	  m = 200
	  n = 200
	  r = 20
	  epsilon = 1e-3
	  #epsilon = 0

	  W = np.random.random((m, r))
	  M = np.dot(W, np.hstack((np.eye(r), self.Hprime)))
	  # permutation of columns
	  P = range(n)
	  for i in xrange(r):
		  P[i], P[i * (n / r)] = P[i * (n / r)], P[i]
	  #P1 = np.arange(0, n, n / r)
	  #P2 = [x for x in np.arange(0, n) if x not in P1]
	  #P = list(P1) + P2
	  M = M[:, P]

	  # Add noise
	  N = np.random.random((m, n)) * epsilon
	  M += N

	  for row in M:
		  key = [np.random.random() for i in xrange(3)]
		  val = [float(v) for v in row]
		  yield key, struct.pack('d' * len(val), *val)

def runner(job):
    options=[('numreducetasks', '0'), ('nummaptasks', '40')]
    job.additer(mapper=Map('Hprime_20_200.txt'), reducer=mrnmf.ID_REDUCER, opts=options)

def starter(prog):
    print "running starter!"

    mypath =  os.path.dirname(__file__)
    print "my path: " + mypath

    # set the global opts
    gopts.prog = prog

    mat = prog.delopt('mat')
    if not mat:
        return "'mat' not specified'"

    prog.addopt('memlimit','8g')

    prog.addopt('file',os.path.join(mypath,'util.py'))
    prog.addopt('file',os.path.join(mypath,'mrnmf.py'))
    prog.addopt('file',os.path.join(mypath,'Hprime_20_200.txt'))

    prog.addopt('input', mat)
    matname,matext = os.path.splitext(mat)

    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-randn%s'%(matname,'.bseq'))

    prog.addopt('overwrite','yes')
    prog.addopt('jobconf','mapred.output.compress=true')

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
