#!/usr/bin/env python

# By Yangyang Hou, based on TypedBytes for python by Klaas Bosteels

# Typed bytes types:
BYTES = 0
BYTE = 1
BOOL = 2
INT = 3
LONG = 4
FLOAT = 5
DOUBLE = 6
STRING = 7
VECTOR = 8
LIST = 9
MAP = 10
# Application-specific types:
PICKLE = 100
BYTESTRING = 101
# Low-level types:
MARKER = 255


def classes():

    from cPickle import dumps, loads, UnpicklingError, HIGHEST_PROTOCOL
    from struct import pack, unpack, error as StructError
    from array import array
   
    from types import BooleanType, IntType, LongType, FloatType 
    from types import UnicodeType, StringType, TupleType, ListType, DictType
    from datetime import datetime, date
    from decimal import Decimal

    UNICODE_ENCODING = 'utf8'

    _len = len


    class Bytes(str):

        def __repr__(self):
            return "Bytes(" + str.__repr__(self) + ")"


    class Input(object):

        def __init__(self, file, unicode_errors='strict'):
            self.file = file
            self.unicode_errors = unicode_errors
            self.eof = False
            self.handler_table = self._make_handler_table()

        def _read(self):
            try:
                t = self.file.readUByte()
                self.t = t
                #print 'I am a type code    ',t
            except StructError:
                self.eof = True
                raise StopIteration
            return self.handler_table[t](self)

        def read(self):
            try:
                return self._read()
            except StopIteration:
                return None

        def _reads(self):
            r = self._read
            while 1:
                yield r()

        __iter__ = reads = _reads

        def close(self):
            self.file.close()

        def read_bytes(self):
            count = self.file.readInt()
            value = self.file.read(count)
            #print 'in read_bytes, len is ', len(value),'value is', value
            if _len(value) != count:
               raise StructError("EOF before reading all of bytes type") 
            return Bytes(value)

        def read_byte(self):
            return self.file.readByte()

        def read_bool(self):
            return bool(self.file.readBoolean())

        def read_int(self):
            return self.file.readInt()

        def read_long(self):
            return self.file.readLong()

        def read_float(self):
            return self.file.readFloat()

        def read_double(self):
            return self.file.readDouble()

        def read_string(self):
            count = self.file.readInt()
            value = self.file.read(count)
            if _len(value) != count:
                raise StructError("EOF before reading all of string")
            return value

        read_bytestring = read_string

        def read_unicode(self):
            count = self.file.readInt()
            value = self.file.read(count)
            if _len(value) != count:
                raise StructError("EOF before reading all of string")
            return value.decode(UNICODE_ENCODING, self.unicode_errors)

        def read_vector(self):
            r = self._read
            count = self.file.readInt()
            try:
                return tuple(r() for i in xrange(count))
            except StopIteration:
                raise StructError("EOF before all vector elements read")

        def read_list(self):
            value = list(self._reads())
            if self.eof:
                raise StructError("EOF before end-of-list marker")
            return value

        def read_map(self):
            r = self._read
            count = self.file.readInt()
            return dict((r(), r()) for i in xrange(count))

        def read_pickle(self):
            count = self.file.readInt()
            bytes = self.file.read(count)
            if _len(bytes) != count:
                raise StructError("EOF before reading all of bytes type")
            return loads(bytes)

        def read_marker(self):
            raise StopIteration

        def invalid_typecode(self):
            raise StructError("Invalid type byte: " + str(self.t))

        TYPECODE_HANDLER_MAP = {
            BYTES: read_bytes,
            BYTE: read_byte,
            BOOL: read_bool,
            INT: read_int,
            LONG: read_long,
            FLOAT: read_float,
            DOUBLE: read_double,
            STRING: read_string,
            VECTOR: read_vector,
            LIST: read_list,
            MAP: read_map,
            PICKLE: read_pickle,
            BYTESTRING: read_bytestring,
            MARKER: read_marker
        }

        def _make_handler_table(self):
            return list(Input.TYPECODE_HANDLER_MAP.get(i,
                        Input.invalid_typecode) for i in xrange(256))

        def register(self, typecode, handler):
            self.handler_table[typecode] = handler

        def lookup(self, typecode):
            return lambda: self.handler_table[typecode](self)


    _BYTES, _BYTE, _BOOL = BYTES, BYTE, BOOL
    _INT, _LONG, _FLOAT, _DOUBLE = INT, LONG, FLOAT, DOUBLE
    _STRING, _VECTOR, _LIST, _MAP = STRING, VECTOR, LIST, MAP
    _PICKLE, _BYTESTRING, _MARKER = PICKLE, BYTESTRING, MARKER

    _int, _type = int, type


    def flatten(iterable):
        for i in iterable:
            for j in i:
                yield j


    class Output(object):

        def __init__(self, file, unicode_errors='strict'):
            self.file = file
            self.unicode_errors = unicode_errors
            self.handler_map = self._make_handler_map()

        def __del__(self):
            if not file.closed:
                self.file.flush()

        def _write(self, obj):
            try:
                writefunc = self.handler_map[_type(obj)]
            except KeyError:
                writefunc = Output.write_pickle
            #print 'I am object   ',obj, 'Type is ',_type(obj)
            writefunc(self, obj)

        write = _write

        def _writes(self, iterable):
            #print 'in _writes', iterable
            w = self._write
            for obj in iterable:
                w(obj)

        writes = _writes

        def flush(self):
            self.file.flush()

        def close(self):
            self.file.close()

        def write_bytes(self, bytes):
            self.file.writeUByte(_BYTES)
            self.file.writeInt(_len(bytes))
            self.file.write(bytes)

        def write_byte(self, byte):
            self.file.writeUByte(_BYTE)
            self.file.writeByte(byte)

        def write_bool(self, bool_):
            self.file.writeUByte(_BOOL)
            self.file.writeByte(_int(bool_))

        def write_int(self, int_):
            # Python ints are 64-bit
            if -2147483648 <= int_ <= 2147483647:
                self.file.writeUByte(_INT)
                self.file.writeInt(int_)
            else:
                self.file.writeUByte(_LONG)
                self.file.writeLong(int_)

        def write_long(self, long_):
            # Python longs are infinite precision
            if -9223372036854775808L <= long_ <= 9223372036854775807L:
                self.file.writeUByte(_LONG)
                self.file.writeLong(long_)
            else:
                self.write_pickle(long_)

        def write_float(self, float_):
            self.file.writeUByte(_FLOAT)
            self.file.writeFloat(float_)

        def write_double(self, double):
            self.file.writeUByte(_DOUBLE)
            self.file.writeDouble(double)

        def write_string(self, string):
            self.file.writeUByte(_STRING)
            self.file.writeInt(_len(string))
            self.file.write(string)

        def write_bytestring(self, string):
            self.file.writeUByte(_BYTESTRING)
            self.file.writeInt(_len(string))
            self.file.write(string)

        def write_unicode(self, string):
            string = string.encode(UNICODE_ENCODING, self.unicode_errors)
            self.file.writeUByte(_STRING)
            self.file.writeInt(_len(string))
            self.file.write(string)

        def write_vector(self, vector):
            self.file.writeUByte(_VECTOR)
            self.file.writeInt(_len(vector))
            #print 'vector length   ',len(vector)
            self._writes(vector)

        def write_list(self, list_):
            self.file.writeUByte(LIST)
            self._writes(list_)
            self.file.writeUByte(MARKER)

        def write_map(self, map):
            self.file.writeUByte(_MAP)
            self.file.writeInt(_len(map))
            self._writes(flatten(map.iteritems()))

        def write_pickle(self, obj):
            bytes = dumps(obj, HIGHEST_PROTOCOL)
            self.file.writeUByte(_PICKLE)
            self.file.writeInt(_len(bytes))
            self.file.write(bytes)

        def write_array(self, arr):
            bytes = arr.tostring()
            self.file.writeUByte(_BYTES)
            self.file.writeInt(_len(bytes))
            self.file.write(bytes)

        TYPE_HANDLER_MAP = {
            BooleanType: write_bool,
            IntType: write_int,                
            LongType: write_long,       
            FloatType: write_double,
            StringType: write_string,
            TupleType: write_vector,
            ListType: write_list,        
            DictType: write_map,
            UnicodeType: write_unicode,
            Bytes: write_bytes,
            datetime: write_pickle,
            date: write_pickle,
            Decimal: write_pickle,
            array: write_array
        }

        def _make_handler_map(self):
            return dict(Output.TYPE_HANDLER_MAP)

        def register(self, python_type, handler):
            self.handler_map[python_type] = handler

        def lookup(self, python_type):
            handler_map = self.handler_map
            if python_type in handler_map:
                return lambda obj: handler_map[python_type](self, obj)
            else:
                return lambda obj: Output.write_pickle(self, obj)
   

    return Input, Output, Bytes


Input, Output, Bytes = classes()