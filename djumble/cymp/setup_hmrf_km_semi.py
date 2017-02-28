from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_module = Extension(
    "hmrf_km_semi",
    ["hmrf_km_semi.pyx"],
    libraries=["m"],
    extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    extra_link_args=['-fopenmp']
)

setup(
    name='hmrf_km_semi module',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
