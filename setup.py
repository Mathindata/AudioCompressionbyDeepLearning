#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="End-to-end Codec Design Toolbox",
    version="1.0.0",
    author="Matin Azadmanesh,Simon Brodeur, ",
    author_email="Azadmanesh.Matin@usherbrooke.ca",
    description=("End-to-end speech compression with perceptual feedback from various priors"),
    license="BSD 3-Clause License",
    keywords="speech compression, machine learning, signal processing",
    url="https://github.com/sbrodeur/speech-compression-asr",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=['setuptools-markdown'],
    install_requires=[
        "setuptools-markdown",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    long_description_markdown_filename='README.md',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
    ],
)
