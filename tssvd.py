#!/usr/bin/env dumbo

"""
tssvd.py
===========

Driver code for NMF column selection using the SVD approach.
Give A, compute A = USV', and let Z = SV'.  The extreme points
in A are the same as the extreme points in Z, since we have
only hit the data points with a linear transformation.  However,
the transformation has introduced many zeros, and we only need
to look at the non-zero region, so we can consider S to be the
square SVD matrix.

Example usage:
     dumbo start tssvd.py -mat A_800M_10.bseq \
     -reduce_schedule 40,1 -hadoop icme-hadoop1


Austin R. Benson (arbenson@stanford.edu)
Copyright (c) 2013
"""

import mrnmf
import dumbo
import util
import os

# create the global options structure
gopts = util.GlobalOptions()

def runner(job):
    blocksize = gopts.getintkey('blocksize')
    rank = gopts.getintkey('rank')
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i, part in enumerate(schedule):
        isfinal = (i == len(schedule) - 1)
        nreducers = int(part)
        mapper = mrnmf.SVDSelect(blocksize=blocksize, isreducer=False,
                                 isfinal=isfinal, rank=rank)
        reducer = mrnmf.SVDSelect(blocksize=blocksize, isreducer=True,
                                  isfinal=isfinal, rank=rank)
        job.additer(mapper=mapper,reducer=reducer,
                    opts = [('numreducetasks', str(nreducers))])    

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    gopts.getintkey('blocksize', 3)
    gopts.getintkey('rank', 6)
    gopts.getstrkey('reduce_schedule', '1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output', '%s-svd%s'%(matname, matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
