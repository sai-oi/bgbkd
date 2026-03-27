"""
setup.py — BGBKD package installation.
Run: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name         = "bgbkd",
    version      = "0.1.0",
    description  = "Battery Graph Bistable Koopman Dictionary — Pack-Level Thermal Runaway Cascade Forecasting",
    author       = "SAI-OI / ROIS · Conception & direction: Mene · Formulation: Claude",
    packages     = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
    ],
    extras_require = {
        "drt":  ["impedance>=1.4.2"],
        "test": ["pytest>=7.4", "pytest-benchmark"],
        "hw":   ["smbus2"],
    },
    entry_points = {
        "console_scripts": [
            "bgbkd-demo=bgbkd.demo.ev_fastcharge:main",
        ],
    },
)
