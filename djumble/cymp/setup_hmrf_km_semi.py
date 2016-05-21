from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize



ext_module = Extension(
    "hmrf_km_semi",
    ["hmrf_km_semi.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='hmrf_km_semi module',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
