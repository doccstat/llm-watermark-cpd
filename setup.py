from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("watermarking.transform.its_levenshtein", [
              "watermarking/transform/its_levenshtein.pyx"]),
    Extension("watermarking.transform.transform_levenshtein", [
              "watermarking/transform/transform_levenshtein.pyx"]),
    Extension("watermarking.gumbel.gumbel_levenshtein", [
              "watermarking/gumbel/gumbel_levenshtein.pyx"]),
    Extension("watermarking.gumbel.ems_levenshtein", [
              "watermarking/gumbel/ems_levenshtein.pyx"]),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
