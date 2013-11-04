#!/usr/bin/env dumbo

"""
Austin R. Benson
Copyright (c) 2013
"""

import dumbo
import mrnmf
import os
import sys
import util

# create the global options structure
gopts = util.GlobalOptions()
    
def runner(job):
    blocksize = gopts.getintkey('blocksize')
    schedule = gopts.getstrkey('reduce_schedule')
    cols_path = gopts.getstrkey('cols_path')
    try:
        f = open(cols_path, 'r')
    except:
        # We may be expecting only the file to be distributed with the script
        f = open(cols_path.split('/')[-1], 'r')
    cols = [int(line.split()[-1]) for line in f]
    cols.sort()
    schedule = schedule.split(',')
    for i,part in enumerate(schedule):
        nreducers = int(part)
        if i == 0:
            mapper = mrnmf.NNLSMapper1(cols=cols, blocksize=blocksize)
            reducer = mrnmf.ArraySumReducer()
        else:
            mapper = mrnmf.ID_MAPPER
            reducer = mrnmf.NNLSReduce(cols=cols)
            nreducers = 1
        job.additer(mapper=mapper, reducer=reducer,
                    opts=[('numreducetasks', str(nreducers))])

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    gopts.getintkey('blocksize',3)
    gopts.getstrkey('reduce_schedule', '1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"

    cols_path = prog.delopt('cols_path')
    if not cols_path:
        return "'cols_path' not specified"
    prog.addopt('file', os.path.join(os.path.dirname(__file__), cols_path))
    gopts.getstrkey('cols_path', cols_path)
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-NNLS1%s'%(matname,matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
