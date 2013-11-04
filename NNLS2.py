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
    wtw_path = gopts.getstrkey('wtw_path')
    mapper = mrnmf.NNLSMapper2(wtw_path)
    reducer = mrnmf.ID_REDUCER
    nreducers = 0
    job.additer(mapper=mapper, reducer=reducer,
                opts=[('numreducetasks', str(nreducers))])

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    
    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"

    wtw_path = prog.delopt('wtw_path')
    if not wtw_path:
        return "'wtw_path' not specified"
    prog.addopt('file', os.path.join(os.path.dirname(__file__), wtw_path))
    gopts.getstrkey('wtw_path', wtw_path)
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output','%s-NNLS2%s'%(matname,matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
