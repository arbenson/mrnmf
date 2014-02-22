#!/usr/bin/env python
# ========================================================================
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from hadoop.io import SequenceFile

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: SequenceFileReader <filename>'
    else:
        reader = SequenceFile.Reader(sys.argv[1])

        key_class = reader.getKeyClass()
        value_class = reader.getValueClass()

        key = key_class()
        value = value_class()

        position = reader.getPosition()
        while reader.next(key, value):
            # sometimes this grabs newlines in the middle that we do not want
            # TODO(arbenson): make this cleaner
            val_list = value.toString().split('\n')
            val_str = ''
            for val in val_list:
                val_str += val
            print '(%s) %s' % (key.toString(), val_str)
            position = reader.getPosition()

        reader.close()

