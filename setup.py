from distutils.core import setup
from setuptools import find_packages

setup(name="implicitresnet",
	  packages=find_packages(),
	  package_dir={'implicitresnet': 'implicitresnet'},
	  author='Viktor Reshniak',
      author_email='reshniakv@ornl.gov',
      version='0.1.2'
      )