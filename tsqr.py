#!/usr/bin/env dumbo

"""
tsqr.py
===========

Driver code for tsqr.

Example usage:
     dumbo start tsqr.py -mat A_800M_10.bseq -nummaptasks 30 \
     -reduce_schedule 20,1 -hadoop icme-hadoop1


Austin R. Benson (arbenson@stanford.edu)
David F. Gleich
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
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i,part in enumerate(schedule):
        isfinal = (i == len(schedule) - 1)
        nreducers = int(part)
        mapper = mrnmf.SerialTSQR(blocksize=blocksize, isreducer=False,
                                 isfinal=isfinal)
        reducer = mrnmf.SerialTSQR(blocksize=blocksize, isreducer=True,
                                  isfinal=isfinal)
        job.additer(mapper=mapper,reducer=reducer,
                    opts = [('numreducetasks', str(nreducers))])    

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    gopts.getintkey('blocksize',3)
    gopts.getstrkey('reduce_schedule','1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output', '%s-qrr%s'%(matname, matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
