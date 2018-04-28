from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# ext_modules=[
#     Extension("fourier",
#               sources=["fourier.pyx"],
#               libraries=["m"] # Unix-like specific
#     )
# ]

setup(
  ext_modules = cythonize("fourier.pyx", annotate = True)
)