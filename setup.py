# setup.py script to build and install patchmatch extension module written in C

from distutils.core      import setup
from Cython.Build        import cythonize


# Note: setup() has access to cmd arguments of the setup.py script via sys.argv
setup(name="implicitresnet",
	  packages=["implicitresnet"],
	  package_dir={'implicitresnet': 'src'},
	  author='Viktor Reshniak',
      author_email='reshniakv@ornl.gov'
      )