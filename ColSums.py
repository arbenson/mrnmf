#!/usr/bin/env dumbo

"""
ColSums.py
===========

Driver code for computing the column sums of a matrix.

Example usage:
     dumbo start ColSums.py -mat A_800M_10.bseq \
     -reduce_schedule 40,1 -hadoop icme-hadoop1


Austin R. Benson (arbenson@stanford.edu)
Copyright (c) 2014
"""

import mrnmf
import dumbo
import util
import os

# create the global options structure
gopts = util.GlobalOptions()

def runner(job):
    schedule = gopts.getstrkey('reduce_schedule')
    
    schedule = schedule.split(',')
    for i, part in enumerate(schedule):
        isfinal = (i == len(schedule) - 1)
        nreducers = int(part)
        if not isfinal:
            mapper = mrnmf.ColSumsMap()
        else:
            mapper = mrnmf.ID_MAPPER
        reducer = mrnmf.ColSumsRed()
        job.additer(mapper=mapper,reducer=reducer,
                    opts = [('numreducetasks', str(nreducers))])    

def starter(prog):
    # set the global opts    
    gopts.prog = prog
    gopts.getstrkey('reduce_schedule','1')

    mat = mrnmf.starter_helper(prog)
    if not mat: return "'mat' not specified"
    
    matname,matext = os.path.splitext(mat)
    output = prog.getopt('output')
    if not output:
        prog.addopt('output', '%s-colsums%s'%(matname, matext))

    gopts.save_params()

if __name__ == '__main__':
    dumbo.main(runner, starter)
