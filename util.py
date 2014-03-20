"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich, 
   Purdue University, and Stanford University.
   All rights reserved.
 
   This file is part of MRNMF and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
"""

import os
import subprocess
import sys
import time

"""
Utility routines NMF and launching MapReduce jobs.
"""

def array2list(row):
    return [float(val) for val in row]

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

    def getfloatkey(self,key,default=None):
        return self._get_key(key,default,float)
            
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
