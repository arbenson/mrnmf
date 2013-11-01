#!/usr/bin/env dumbo

"""
Austin R. Benson (arbenson@stanford.edu)
Copyright (c) 2013
"""

import sys
import os
import time
import random
import struct

import numpy as np
#from cvxopt import matrix, solvers

import util
import dumbo
import dumbo.backends.common

from dumbo import opt

# some variables
ID_MAPPER = 'org.apache.hadoop.mapred.lib.IdentityMapper'
ID_REDUCER = 'org.apache.hadoop.mapred.lib.IdentityReducer'

class DataFormatException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def starter_helper(prog, use_dirtsqr=False, use_house=False):
    print 'running starter!'

    mypath = os.path.dirname(__file__)
    print 'my path: ' + mypath    

    prog.addopt('file', os.path.join(mypath, 'util.py'))
    prog.addopt('file', os.path.join(mypath, 'mrnmf.py'))

    splitsize = prog.delopt('split_size')
    if splitsize is not None:
        prog.addopt('jobconf',
            'mapreduce.input.fileinputformat.split.minsize=' + str(splitsize))

    prog.addopt('overwrite', 'yes')
    prog.addopt('jobconf', 'mapred.output.compress=true')
    prog.addopt('memlimit', '8g')

    mat = prog.delopt('mat')
    if mat:
        # add numreps copies of the input
        numreps = prog.delopt('repetition')
        if not numreps:
            numreps = 1
        for i in range(int(numreps)):
            prog.addopt('input',mat)
    
        return mat            
    else:
        return None


"""
MatrixHandler reads data and collects it
"""
class MatrixHandler(dumbo.backends.common.MapRedBase):
    def __init__(self):
        self.ncols = None
        self.unpacker = None
        self.nrows = 0
        self.deduced = False

    def collect(self, key, value):
        pass

    def collect_data_instance(self, key, value):
        if isinstance(value, str):
            if not self.deduced:
                self.deduced = self.deduce_string_type(value)
                # handle conversion from string
            if self.unpacker is not None:
                value = self.unpacker.unpack(value)
            else:
                value = [float(p) for p in value.split()]

        if self.ncols == None:
            self.ncols = len(value)
            print >>sys.stderr, 'Matrix size: %i columns' % (self.ncols)
        if len(value) != self.ncols:
            raise DataFormatException(
                'Length of value did not match number of columns')
        self.collect(key, value)        

    def collect_data(self, data, key=None):
        if key == None:
            for key, value in data:
                self.collect_data_instance(key, value)
        else:
            for value in data:
                self.collect_data_instance(key, value)

    def deduce_string_type(self, val):
        # first check for TypedBytes list/vector
        try:
            [float(p) for p in val.split()]
        except:
            if len(val) == 0:
                return False
            if len(val) % 8 == 0:
                ncols = len(val) / 8
                # check for TypedBytes string
                try:
                    val = list(struct.unpack('d' * ncols, val))
                    self.unpacker = struct.Struct('d' * ncols)
                    return True
                except struct.error, serror:
                    # no idea what type this is!
                    raise DataFormatException('Data format type is not supported.')
            else:
                raise DataFormatException('Number of data bytes (%d)' % len(val)
                                          + ' is not a multiple of 8.')
        return True

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

# Mapper that reads in rows of the matrix and computes NMF on blocks
class SerialNMF(MatrixHandler):
    def __init__(self, r=8, epsilon=1e-5, blocksize=12):
        MatrixHandler.__init__(self)
        self.r = r
        self.epsilon = epsilon
        self.blocksize = blocksize
        self.data = []
        self.votes = {}
    
    def NMF(self):
        M = np.array(data)
        return HotTopixx(M, epsilon, r)
    
    def compress(self):
        # Compute NMF on the data accumulated so far.
        if self.ncols is None:
            return

        t0 = time.time()
        cols = self.NMF()
        dt = time.time() - t0
        self.counters['numpy time (millisecs)'] += int(1000 * dt)

        # Re-initialize
        self.data = []
        for col in columns:
            self.votes[col] += 1
                        
    def collect(self, key, value):
        self.data.append(value)
        self.nrows += 1
        
        if len(self.data) > self.blocksize * self.ncols:
            self.counters['NMF compressions'] += 1
            # compress the data
            self.compress()
            
        # write status updates so Hadoop doesn't complain
        if self.nrows % 50000 == 0:
            self.counters['rows processed'] += 50000

    def close(self):
        self.counters['rows processed'] += self.nrows % 50000
        self.compress()
        for vote in self.votes:
            yield vote, self.votes[vote]

    def __call__(self, data):
        self.collect_data(data)

        # finally, output data
        for key, val in self.close():
            yield key, val

# Reducer that just sums votes
class MajorityVotes():
    def __init__(self, r):
        self.votes = {}
        self.r = r

    def close(self):
        # Return top r vote-getting columns
        total = max(self.r, len(self.votes))
        return sorted(self.votes.items(), key=lambda x: x[1], reverse=True)[0:total]

    def __call__(self, data):
        for key, values in data:
            for val in values:
                self.votes[key] += val

        for key, val in self.close():
            yield key, val

class GaussianReduction(MatrixHandler):
    def __init__(self, blocksize=5, projsize=200):
        MatrixHandler.__init__(self)
        self.blocksize = blocksize
        self.data = []
        self.projsize = projsize
        self.A_curr = None
    
    def compress(self):
        if self.ncols is None or len(self.data) == 0:
            return

        t0 = time.time()
        G = np.random.randn(self.projsize, len(self.data)) / 10.
        A_flush = G * np.mat(self.data)
        dt = time.time() - t0
        self.counters['numpy time (millisecs)'] += int(1000 * dt)

        # Add flushed update to local copy
        if self.A_curr == None:
            self.A_curr = A_flush
        else:
            self.A_curr += A_flush
        self.data = []
                        
    def collect(self, key, value):
        self.data.append(value)
        self.nrows += 1
        
        if len(self.data) > self.blocksize * self.ncols:
            self.counters['Gaussian compressions'] += 1
            # compress the data
            self.compress()
            
        # write status updates so Hadoop doesn't complain
        if self.nrows % 50000 == 0:
            self.counters['rows processed'] += 50000

    def close(self):
        self.counters['rows processed'] += self.nrows % 50000
        self.compress()
        if self.A_curr is not None:
            for ind, row in enumerate(self.A_curr.getA()):
                yield ind, util.array2list(row)

    def __call__(self, data):
        self.collect_data(data)

        # finally, output data
        for key, val in self.close():
            yield key, val

class ArraySumReducer(MatrixHandler):
    def __init__(self):
        MatrixHandler.__init__(self)
        self.row_sums = {}

    def collect(self, key, value):
        if key not in self.row_sums:
            self.row_sums[key] = value
        else:
            if len(value) != len(self.row_sums[key]):
                print >>sys.stderr, 'value: ' + str(value)
                print >>sys.stderr, 'value: ' + str(self.row_sums[key])
                raise DataFormatException('Differing array lengths for summing')
            for k in xrange(len(self.row_sums[key])):
                self.row_sums[key][k] += value[k]
    
    def __call__(self, data):
        for key, values in data:
            self.collect_data(values, key)
        for key in self.row_sums:
            yield key, self.row_sums[key]

class ProjectionReducer():
    def __init__(self):
        self.data = []
        self.cols = set()

    def close(self):
        for row in self.data:
            self.cols.add(row.index(min(row)))
            self.cols.add(row.index(max(row)))
        for col in self.cols:
            yield np.random.rand() * 10000, col
        

    def __call__(self, data):
        for key, values in data:
            for val in values:
                self.data.append(val)

        for key, val in self.close():
            yield key, val
