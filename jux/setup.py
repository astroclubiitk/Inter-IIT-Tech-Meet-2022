from subprocess import list2cmdline
from setuptools import setup

setup(
    name="jux",
    version="0.0.1",
    description="xray flare burst deterministic detection",
    long_description="A library that has deterministic algorithms implemented to detect the features of xray bursts in a light curve.",
    author="Team 10",
    packages=[".jux"],
    install_requires=[
        "astropy",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "plotly",
        "sklearn",
    ],
    liscense="MIT",
    py_modules=[
        "file_handler",
        "create_df_minmax",
        "denoise",
        "false_positive_detection",
        "flare_detect_minmax",
        "flare_detect_thresh",
        "helper",
        "param",
        "version",
        "jux",
    ],
)
