from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("watermarking.transform.transform_levenshtein", [
              "watermarking/transform/transform_levenshtein.pyx"]),
    Extension("watermarking.gumbel.gumbel_levenshtein", [
              "watermarking/gumbel/gumbel_levenshtein.pyx"]),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
