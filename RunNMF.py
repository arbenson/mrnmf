#!/usr/bin/env dumbo

"""
Driver script for NMF.

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
	compute_colsums = gopts.getboolkey('colsums')
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
                                  compute_colsums=True)
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
	gopts.getboolkey('colsums', True)
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
