from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("hmrf_semisup_km_narray_cy.pyx")
)
