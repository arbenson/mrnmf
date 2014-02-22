#!/usr/bin/env python

from distutils.core import setup

setup(name='Hadoop',
      version='0.2',
      description='Python Hadoop I/O Utilities',
      license="Apache Software License 2.0 (ASF)",
      author='Matteo Bertozzi & Yangyang Hou',
      author_email='theo.bertozzi@gmail.com & hyy.sun@gmail.com',
      url='http://hadoop.apache.org',
      packages=["hadoop", 'hadoop.typedbytes2', 'hadoop.util', 'hadoop.io', 'hadoop.io.compress']
     )

