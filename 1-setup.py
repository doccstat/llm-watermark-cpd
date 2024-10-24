from setuptools import Extension

from Cython.Build import cythonize
from numpy import get_include
from setuptools import setup

extensions = [
    Extension("watermarking.transform.transform_levenshtein", [
              "watermarking/transform/transform_levenshtein.pyx"],
              extra_compile_args=['-std=c99']),
    Extension("watermarking.gumbel.gumbel_levenshtein", [
              "watermarking/gumbel/gumbel_levenshtein.pyx"],
              extra_compile_args=['-std=c99']),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[get_include()]
)
