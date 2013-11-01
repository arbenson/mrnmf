"""
util.py
=======

Utility routines for the tsqr and regression code.
"""

import os
import subprocess
import sys
import time

def array2list(row):
    return [float(val) for val in row]

""" A utility to flatten a lists-of-lists. """
def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def parse_matrix_txt(mpath):
    try:
        f = open(mpath, 'r')
    except:
        # We may be expecting only the file to be distributed
        # with the script
        f = open(mpath.split('/')[-1], 'r')        
    data = []
    for line in f:
        ind = line.rfind(')')
        if ind != -1:
            line = line[ind+1:]
        line = line.strip().rstrip().lstrip('[').rstrip(']')
        line2 = line.split(',')
        if len(line2) == 1:
            line2 = line.split()
        line2 = [float(v) for v in line2]
        yield line2

    f.close()

class GlobalOptions:
    """ A class to manage passing options to the actual jobs that run. 
    
    If it's constructed using the option prog, then the class loads 
    options from the dumbo command line parameters.  In this case,
    the class also sets the corresponding environment variable.
    
    Otherwise, it pulls the options from the environment.
    
    Todo: make this have a nicer interface.
    """
    
    def __init__(self,prog=None):
        """ 
        @param prog if prog is specified, then this class sets all
        the parameters, rather than pulling them from the environment.
        """
        self.prog = prog
        self.cache = {}
        
    def _get_key(self,key,default,typefunc):
        if key in self.cache:
            return typefunc(self.cache[key])
        
        if self.prog:
            val = self.prog.delopt(key)
        else:
            val = os.getenv(key)
        
        if val is None:
            if default is None:
                raise NameError(
                    "option '"+key+"' is not a command line "+
                    "or environment option with no default")
            val = default
        else:
            val = typefunc(val)
        
        self.setkey(key,val)
        return val
        
    def getstrkey(self,key,default=None):
        return self._get_key(key,default,str)

    def getintkey(self,key,default=None):
        return self._get_key(key,default,int)        
            
    def setkey(self,key,value):
        if self.prog:
            os.putenv(key,str(value))
            
        self.cache[key] = value
            
    def save_params(self):
        """ This saves all the options to dumbo params. 
        
        For an option to be saved here, it must have been
        "get" or "set" with the options here.
        """
        
        assert(self.prog is not None)
        for key,value in self.cache.items():
            self.prog.addopt('param',str(key)+'='+str(value))
        

"""
CommandManager is our cheapy build system.
"""
class CommandManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.times = []
        self.split = '-'*60

    # simple wrapper around printing with verbose option
    def output(self, msg, split=False):
        if self.verbose:
            if split:
                print self.split
            print msg
            if split:
                print self.split

     # print error messages and exit with failure
    def error(self, msg, code=1):
        print msg
        sys.exit(code)

    def exec_cmd(self, cmd):
        self.output('(command is: %s)' % (cmd), True)
        t0 = time.time()
        retcode = subprocess.call(cmd, shell=True)
        self.times.append(time.time() - t0)
        # TODO(arbenson): make it more obvious when something fails
        return retcode

    # simple wrapper for running dumbo scripts with options provided as a list
    def run_dumbo(self, script, hadoop='', opts=[]):
        cmd = 'dumbo start ' + script
        if hadoop != '':
            cmd += ' -hadoop '
            cmd += hadoop
        for opt in opts:
            cmd += ' '
            cmd += opt
        self.exec_cmd(cmd)

    def copy_from_hdfs(self, inp, outp, delete=True):
        if delete and os.path.exists(outp):
          os.remove(outp)
        copy_cmd = 'hadoop fs -copyToLocal ' \
                   + '%s/part-00000 %s' % (inp, outp)
        self.exec_cmd(copy_cmd)

    def copy_to_hdfs(self, inp, outp):
        copy_cmd = 'hadoop fs -copyFromLocal %s %s' % (inp, outp)
        self.exec_cmd(copy_cmd)

    # parse a sequence file
    def parse_seq_file(self, inp, output=None):
        path = os.path.dirname(__file__)
        # TODO(arbenson): import the files instead of this hack
        reader = os.path.join(path,
                              'hyy-python-hadoop/examples/SequenceFileReader.py')
        if not os.path.exists(reader):
            self.error('Could not find sequence file reader!')
        if output is None:
            output = inp + '.out'
        parse_cmd = 'python %s %s > %s' % (reader, inp, output)
        self.exec_cmd(parse_cmd)

