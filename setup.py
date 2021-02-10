# setup.py script to build and install patchmatch extension module written in C

from distutils.core import setup

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv
setup(name="implicitresnet",
	  packages=["implicitresnet"],
	  package_dir={'implicitresnet': 'implicitresnet'},
	  author='Viktor Reshniak',
      author_email='reshniakv@ornl.gov',
      version='0.1.1'
      )