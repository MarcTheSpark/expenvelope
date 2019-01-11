import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="expenvelope",
    version="0.1.0",
    author="Marc Evanstein",
    author_email="marc@marcevanstein.com",
    description="Provides musically expressive piecewise-exponential curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcTheSpark/expenvelope",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
