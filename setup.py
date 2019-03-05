#!/usr/bin/env python 

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='stepclass',
      version='0.1',
      author='Johnny Greco',
      author_email='jgreco.astro@gmail.com',
      packages=['stepclass'],
      url='https://github.com/johnnygreco/stepclass')
