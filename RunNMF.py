"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich, 
   Purdue University, and Stanford University.
   All rights reserved.
 
   This file is part of MRNMF and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
"""

import dumbo
import os
import util
import mrnmf

"""
Driver script for NMF.  Let X be the data matrix.  This script can compute
any subset of the following:

     1. Gaussian projection: G * X
     2. TSQR: the R factor in X = QR
     3. Column l1 norms: | X(:, i) |_1 for each column index i

Assuming X is nonnegative, the column l1 norms are just the column sums.
All of these properties are computed in one pass over the data.

Usage:

      dumbo start RunNMF.py \
      -libjar feathers.jar \
      -gp [0/1 on whether or not to compute Gaussian projection]
      -projsize [number of rows to project on for GP]
      -qr [0/1 on whether or not to compute R factor in QR factorization] \
      -colnorms [0/1 on whether or not to compute the column l1 norms] \
      -hadoop $HADOOP_INSTALL \
      -reduce_schedule [comma separated list of number of reducers] \
      -mat [name of data file] \
      -output [name of output file] \
      -blocksize [integer block size]

By default, the gp, qr, and colnorms options are True.
The block size specifies how much data to read before compressing the QR
factorization of the Gaussian projection.  If X has n columns and the
blocksize is b, then compression occurs every nb rows.

Example usage for the script is:

      dumbo start RunNMF.py -libjar feathers.jar -projsize 100 \
      -hadoop $HADOOP_INSTALL -reduce_schedule 40,1 \
      -mat cells-kron-40k.bseq -output FC_out.bseq -blocksize 10

This computes (1), (2) and (3) from above for the data matrix
'cells-kron-40k.bseq'.  Alternatively, we can omit the computation
of the R factor in QR:

      dumbo start RunNMF.py -libjar feathers.jar -projsize 100 \
      -hadoop $HADOOP_INSTALL -reduce_schedule 40,1 \
      -mat cells-kron-40k.bseq -output FC_out.bseq -blocksize 10 \
      -qr 0
"""

# create the global options structure
gopts = util.GlobalOptions()

    
def runner(job):
    blocksize = gopts.getintkey('blocksize')
    compute_GP = bool(gopts.getintkey('gp'))
    compute_QR = bool(gopts.getintkey('qr'))
    compute_colnorms = bool(gopts.getintkey('colnorms'))
    projsize = gopts.getintkey('projsize')
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i,part in enumerate(schedule):
        nreducers = int(part)
        if i == 0:
            mapper = mrnmf.NMFMap(blocksize=blocksize,
                                  projsize=projsize,
                                  compute_GP=compute_GP,
                                  compute_QR=compute_QR,
                                  compute_colnorms=compute_colnorms)
            reducer = mrnmf.NMFReduce(blocksize=blocksize,
                                      isfinal=False)
        else:
            mapper = mrnmf.ID_MAPPER
            reducer = mrnmf.NMFReduce(blocksize=blocksize,
                                      isfinal=True)
            nreducers = 1
        job.additer(mapper=mapper, reducer=reducer,
                    opts=[('numreducetasks', str(nreducers))])
    
    # Add a pass where we just separate the data
    job.additer(mapper=mrnmf.ID_MAPPER,
                reducer=mrnmf.NMFParse(),
                opts=[('numreducetasks', '1')])


def starter(prog):
    # set the global opts    
    gopts.prog = prog
    gopts.getintkey('blocksize', 3)
    gopts.getintkey('gp', 1)
    gopts.getintkey('qr', 1)
    gopts.getintkey('colnorms', 1)
    gopts.getintkey('projsize', 400)
    gopts.getstrkey('reduce_schedule', '1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-proj%s' % (matname, matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
