#!/usr/bin/env dumbo

"""
Austin R. Benson (arbenson@stanford.edu)
Copyright (c) 2014
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
try:
    from cvxopt import matrix, solvers
except:
    print >>sys.stderr, 'Missing cvxopt'


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
                '/home/arbenson/hadoop_env/numpy-1.8.0/dist/numpy-1.8.0-py2.6-linux-x86_64.egg',
                '/home/arbenson/hadoop_env/cvxopt/dist/cvxopt-1.1.6-py2.6-linux-x86_64.egg']:
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
            print >>sys.stderr, 'its a string'
            if not self.deduced:
                self.deduced = self.deduce_string_type(value)
                # handle conversion from string
            if self.unpacker is not None:
                value = self.unpacker.unpack(value)
            else:
                value = [float(p) for p in value.split()]
        # check for numpy 2d array
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
        # check for list of lists
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                if self.ncols == None:
                    self.ncols = len(value[0])
                    print >>sys.stderr, 'Matrix size: %i columns' % (self.ncols)
                if len(value[0]) != self.ncols:
                    raise DataFormatException(
                        'Number of columns in value did not match number of columns in matrix')
                for row in value:
                    self.collect_data_instance(key, row)
                return

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

class GaussianProjection(MatrixHandler):
    def __init__(self, blocksize=5, projsize=400):
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

class SerialTSQR(MatrixHandler):
    def __init__(self, blocksize=3, isreducer=False, isfinal=False, rank=6):
        MatrixHandler.__init__(self)
        self.blocksize = blocksize
        self.isreducer = isreducer
        self.isfinal = isfinal
        self.data = []
    
    def QR(self):
        return np.linalg.qr(np.array(self.data),'r')
    
    def compress(self):
        # Compute a QR factorization on the data accumulated so far.
        if self.ncols == None or len(self.data) < self.ncols:
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

    def multicollect(self, key, value):
        """ Collect multiple rows at once with a single key. """
        nkeys = len(value)
        newkey = ('multi', nkeys, key)
        
        self.keys.append(newkey)
        
        for row in value:
            self.add_row(row.tolist())

    def close(self):
        self.counters['rows processed'] += self.nrows % 50000
        self.compress()

        if self.isreducer and self.isfinal:
            for i, row in enumerate(self.data):
                yield i, row
        else:
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
        if self.ncols == None or len(self.data) == 0:
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

