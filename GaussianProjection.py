#!/usr/bin/env dumbo

"""
Austin R. Benson
Copyright (c) 2013
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
    rank = gopts.getintkey('rank')
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i,part in enumerate(schedule):
        nreducers = int(part)
        if i == 0:
            mapper = mrnmf.GaussianReduction(blocksize=blocksize)
            reducer = mrnmf.ArraySumReducer()
        else:
            mapper = mrnmf.ID_MAPPER
            # TODO: Make the target rank an argument
            reducer = mrnmf.ProjectionReducer(rank)
            nreducers = 1
        job.additer(mapper=mapper, reducer=reducer,
                    opts=[('numreducetasks', str(nreducers))])

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    gopts.getintkey('blocksize', 3)
    gopts.getintkey('rank', 10)
    gopts.getstrkey('reduce_schedule', '1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-proj%s'%(matname,matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
