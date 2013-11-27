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

#from cvxopt import matrix, solvers

import util
import dumbo
import dumbo.backends.common
from dumbo import opt

# TODO (arbenson): This is a total hack.
os.environ['PYTHON_EGG_CACHE'] = 'egg_cache'
import numpy as np
try:
    from scipy import optimize
except:
    print >>sys.stderr, 'Missing SciPy'

# some variables
ID_MAPPER = 'org.apache.hadoop.mapred.lib.IdentityMapper'
ID_REDUCER = 'org.apache.hadoop.mapred.lib.IdentityReducer'

class DataFormatException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def starter_helper(prog):
    print 'running starter!'

    mypath = os.path.dirname(__file__)
    print 'my path: ' + mypath    

    prog.addopt('file', os.path.join(mypath, 'util.py'))
    prog.addopt('file', os.path.join(mypath, 'mrnmf.py'))

    for egg in ['/home/arbenson/hadoop_env/scipy-0.13.0/dist/scipy-0.13.0-py2.6-linux-x86_64.egg',
                '/home/arbenson/hadoop_env/numpy-1.8.0/dist/numpy-1.8.0-py2.6-linux-x86_64.egg']:
        prog.addopt('libegg', egg)
        prog.addopt('file', egg)

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
        elif isinstance(value, np.ndarray):
            # verify column size
            if value.ndim == 2:
                # it's a block
                if self.ncols == None:
                    self.ncols = value.shape[1]
                if value.shape[1] != self.ncols:
                    raise DataFormatException(
                        'Number of columns in value did not match number of columns in matrix')
                for row in value:
                    row = row.tolist()
                    self.collect_data_instance(key, row)
                return
            else:
                value = value.tolist() # convert and continue below

        if self.ncols == None:
            self.ncols = len(value)
            print >>sys.stderr, 'Matrix size: %i columns' % (self.ncols)
        if len(value) != self.ncols:
            raise DataFormatException(
                'Length of value did not match number of columns')
        self.collect(key, value)

        
    """
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
    """

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

def HottTopixx(M, epsilon, r):
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
        return HottTopixx(M, epsilon, r)
    
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
        G = np.random.randn(self.projsize, len(self.data)) / 100.
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

class NNLSMapper1(MatrixHandler):
    def __init__(self, cols, blocksize=5):
        MatrixHandler.__init__(self)
        self.blocksize = blocksize
        self.cols = cols
        self.cols.sort()
        self.data = []
        self.A_curr = None
    
    def compress(self):
        if self.ncols is None or len(self.data) == 0:
            return

        t0 = time.time()
        A_mat = np.mat(self.data)
        A_flush = A_mat.T * A_mat[:, self.cols]
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

@opt("getpath", "yes")
class NNLSReduce(MatrixHandler):
    def __init__(self, cols):
        MatrixHandler.__init__(self)
        self.row_sums = {}
        self.cols = cols
        self.cols.sort()

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

    def close(self):
        # We need to emit:
        #    (1) A_i^TW as many key-value pairs
        #    (2) W^TW as a single key-value pair

        # First, the row sums
        for i, row in enumerate(self.row_sums):
            yield ("RHS", i), self.row_sums[row]
        
        # We need need output W^TW with the correct permutation
        WTW = []
        for i in self.cols:
            WTW.append(self.row_sums[i])

        yield ("WTW", -1), WTW
    
    def __call__(self, data):
        for key, values in data:
            self.collect_data(values, key)
        for key, val in self.close():
            yield key, val

@opt("getpath", "yes")
class NNLSMapper2(MatrixHandler):
    def __init__(self, WTW_path):
        MatrixHandler.__init__(self)
        self.parse_WTW(WTW_path)
        self.data = []

    def parse_WTW(self, WTW_path):
        self.WTW = []
        try:
            f = open(WTW_path, 'r')
        except:
            # We may be expecting only the file to be distributed with the script
            f = open(WTW_path.split('/')[-1], 'r')
        mat = f.read()
        f.close()
        for row in mat[mat.rfind(')')+1:].strip().split('],'):
            row = [float(v.rstrip(']')) \
                       for v in row[row.rfind('[') + 1:row.rfind(']')].split(',')]
            self.WTW.append(row)
        self.WTW = np.array(self.WTW)
        
    def collect(self, key, value):
        sol, res = optimize.nnls(self.WTW, value)
        self.data.append((("H", key), sol))
        self.data.append((("Residual", key), res))
            
    def __call__(self, data):
        self.collect_data(data)

        # finally, output data
        for key, val in self.data:
            yield key, val


class SVDSelect(MatrixHandler):
    def __init__(self, blocksize=3, isreducer=False, isfinal=False):
        MatrixHandler.__init__(self)
        self.blocksize = blocksize
        self.isreducer = isreducer
        self.isfinal = isfinal
        self.data = []
    
    def QR(self):
        return np.linalg.qr(np.array(self.data),'r')
    
    def compress(self):
        # Compute a QR factorization on the data accumulated so far.
        if self.ncols is None or len(self.data) < self.ncols:
            return

        t0 = time.time()
        R = self.QR()
        dt = time.time() - t0
        self.counters['numpy time (millisecs)'] += int(1000 * dt)

        # reset data and re-initialize to R
        self.data = []
        for row in R:
            self.data.append(util.array2list(row))
                        
    def collect(self, key, value):
        self.data.append(value)
        self.nrows += 1
        
        if len(self.data) > self.blocksize * self.ncols:
            self.counters['QR Compressions'] += 1
            self.compress()
            
        # write status updates so Hadoop doesn't complain
        if self.nrows % 50000 == 0:
            self.counters['rows processed'] += 50000

    def compute_extreme_pts(self):
        _, S, V = np.linalg.svd(np.array(self.data))
        self.data = []
        cols = set()
        for row in S * V:
            cols.add(np.argmax(row))
            cols.add(np.argmin(row))
        for col in cols:
            self.data.append(col)

    def multicollect(self, key, value):
        """ Collect multiple rows at once with a single key. """
        nkeys = len(value)
        newkey = ('multi',nkeys,key)
        
        self.keys.append(newkey)
        
        for row in value:
            self.add_row(row.tolist())

    def close(self):
        self.counters['rows processed'] += self.nrows % 50000
        self.compress()
        if self.isreducer and self.isfinal:
            self.compute_extreme_pts()
        for i, row in enumerate(self.data):
            key = np.random.randint(0, 4000000000)
            yield key, row

    def __call__(self, data):
        if not self.isreducer:
            self.collect_data(data)
        else:
            for key, values in data:
                self.collect_data(values, key)

        for key, val in self.close():
            yield key, val

class ColSumsMap(MatrixHandler):
    def __init__(self):
        MatrixHandler.__init__(self)
        self.data = {}
    
    def collect(self, key, value):
        for col, v in enumerate(value):
            if col in self.data:
                self.data[col] += v
            else:
                self.data[col] = v


    def __call__(self, data):
        self.collect_data(data)

        for key in self.data:
            yield key, self.data[key]

class ColSumsRed():
    def __init__(self):
        pass

    def __call__(self, data):
        for key, values in data:
            yield key, sum(values)

class ColScale(MatrixHandler):
    def __init__(self, cols, blocksize=3):
        MatrixHandler.__init__(self)        
        self.blocksize = blocksize
        self.data = []
        self.keys = []
        self.small = np.mat(np.linalg.inv(np.diag(cols)))

    def compress(self):        
        # Compute the matmul on the data accumulated so far
        if self.ncols is None or len(self.data) == 0:
            return

        self.counters['MatMul compression'] += 1

        t0 = time.time()
        A = np.mat(self.data)
        out_mat = A * self.small
        dt = time.time() - t0
        self.counters['numpy time (millisecs)'] += int(1000 * dt)

        # reset data and add flushed update to local copy
        self.data = []
        for i, row in enumerate(out_mat.getA()):
            yield self.keys[i], struct.pack('d' * len(row), *row)

        # clear the keys
        self.keys = []
    
    def collect(self, key, value):
        self.keys.append(key)
        self.data.append(value)
        self.nrows += 1
        
        # write status updates so Hadoop doesn't complain
        if self.nrows%50000 == 0:
            self.counters['rows processed'] += 50000

    def __call__(self, data):
        for key,value in data:
            self.collect_data_instance(key, value)

            # if we accumulated enough rows, output some data
            if len(self.data) >= self.blocksize * self.ncols:
                for key, val in self.compress():
                    yield key, val
                    
        # output data the end of the data
        for key, val in self.compress():
            yield key, val

