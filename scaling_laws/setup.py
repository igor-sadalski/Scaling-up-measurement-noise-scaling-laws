from setuptools import setup

setup(
    name="scaling_laws",
    version="0.0.1",
    description="Scaling Laws for Single-Cell Data",
    packages=["scaling_laws"],
    include_package_data=True,
    install_requires=[
        "datasets",
        "loompy",
        "numpy",
        "tdigest",
        "transformers",
    ],
)
