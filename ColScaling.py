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
    sums = gopts.getstrkey('sums')
    try:
        f = open(sums, 'r')
    except:
        # We may be expecting only the file to be distributed with the script
        f = open(sums.split('/')[-1], 'r')
    cols = [(int(line.split()[0].strip('()')), float(line.split()[-1])) for line in f]
    cols = [y[1] for y in sorted(cols, key=lambda x: x[0])]
    schedule = schedule.split(',')
    mapper = mrnmf.ColScale(cols=cols, blocksize=blocksize)
    reducer = mrnmf.ID_REDUCER
    job.additer(mapper=mapper, reducer=reducer,
                opts=[('numreducetasks', str(0))])

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    gopts.getintkey('blocksize',3)
    gopts.getstrkey('reduce_schedule', '1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"

    sums = prog.delopt('sums')
    if not sums:
        return "'sums' not specified"
    prog.addopt('file', os.path.join(os.path.dirname(__file__), sums))
    gopts.getstrkey('sums', sums)
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-scaled%s'%(matname,matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
