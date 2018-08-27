import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setuptools.setup(
    name="Djumble",
    version="0.0.1",
    author="Dimitrios Pritsos",
    author_email="dpritsos@extremepro.gr",
    description="Semi-supervised Clustering with Constraint Expectation Maximization (Kmeans)",
    long_description="",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)

ext_modules = [
    Extension(
        "Djumble/djumble/voperators/cy",
        ["Djumble/djumble/voperators/cy.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    setup_requires=[
        'setuptools>=18.0',
        'cython>=0.19.1',
    ],
    name='cy',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
