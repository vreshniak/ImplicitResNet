# setup.py script to build and install patchmatch extension module written in C

from distutils.core import setup
from setuptools import find_packages

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv
setup(name="implicitresnet",
	  packages=find_packages(),
	  package_dir={'implicitresnet': 'implicitresnet'},
	  author='Viktor Reshniak',
      author_email='reshniakv@ornl.gov',
      version='0.1.1'
      )