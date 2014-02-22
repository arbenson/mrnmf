#!/usr/bin/env python

from hadoop.io.Writable import AbstractValueWritable
from hadoop.typedbytes2 import typedbytes2
from hadoop.io.OutputStream import DataOutputBuffer

class TypedBytesWritable(AbstractValueWritable):
    def write(self, data_output):
        tmpout = DataOutputBuffer()
        output = typedbytes2.Output(tmpout)
        output.write(self._value)
        bytes_len = tmpout.getSize()
        #print 'first 4 bytes in data output is ',bytes_len
        data_output.writeInt(bytes_len)
        data_output.write(tmpout.toByteArray())
        
    def readFields(self, data_input):
        # print "in read fields!"
        bytes_len = data_input.readInt()
        #print 'first 4 bytes in data input is ',bytes_len
        input = typedbytes2.Input(data_input)
        for record in input:
            self._value = record
