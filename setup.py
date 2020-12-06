import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nld",
    version="0.0.1",
    author="Andrea Favia",
    author_email="drewlinguistics01@gmail.com",
    description="A small package that provides decorators for text preprocessing with nltk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andcarnivorous/nld",
    packages=setuptools.find_packages(),
    install_requires=["nltk==3.4.5", "pandas>=0.25.3", "numpy==1.19.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)