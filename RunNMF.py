#!/usr/bin/env dumbo

"""
Driver script for NMF.  Let X be the data matrix.  This script can compute
any subset of the following:

     1. Gaussian projection: G * X
     2. TSQR: the R factor in X = QR
     3. Column l1 norms: | X(:, i) |_1 for each column index i

Assuming X is nonnegative, the column l1 norms are just the column sums.
All of these properties are computed in one pass over the data.

The script call is:

      dumbo start RunNMF.py \
      -libjar feathers.jar \
      -gp [0/1 on whether or not to compute Gaussian projection]
      -qr [0/1 on whether or not to compute R factor in QR factorization] \
      -colnorms [0/1 on whether or not to compute the column l1 norms] \
      -hadoop [name of Hadoop for Dumbo to use] \
      -reduce_schedule [comma separated list of number of reducers] \
      -mat [name of data file] \
      -output [name of output file] \
      -blocksize [integer block size]

By default, the gp, qr, and colnorms options are True.
The block size specifies how much data to read before compressing the QR
factorization of the Gaussian projection.  If X has n columns and the
blocksize is b, then compression occurs every nb rows.

Example usage for the script is:

      dumbo start RunNMF.py -libjar feathers.jar -hadoop icme-hadoop1 -reduce_schedule 40,1 \
      -mat cells-kron-40k.bseq -output FC_out.bseq -blocksize 10

This computes (1), (2) and (3) from above for the FC data on
ICME's Hadoop cluster.
     

Austin R. Benson
Copyright (c) 2014
"""

import dumbo
import os
import sys
import util
import mrnmf


# create the global options structure
gopts = util.GlobalOptions()

    
def runner(job):
    blocksize = gopts.getintkey('blocksize')
    compute_GP = gopts.getboolkey('gp')
    compute_QR = gopts.getboolkey('qr')
    compute_colnorms = gopts.getboolkey('colnorms')
    projsize = gopts.getintkey('projsize')
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i,part in enumerate(schedule):
        nreducers = int(part)
        if i == 0:
            mapper = mrnmf.NMFMap(blocksize=blocksize,
                                  projsize=projsize,
                                  compute_GP=True,
                                  compute_QR=True,
                                  compute_colnorms=True)
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
    gopts.getboolkey('gp', True)
    gopts.getboolkey('qr', True)
    gopts.getboolkey('colnorms', True)
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
