import setuptools

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="neurolib",
    version="0.5.12",
    description="Easy whole-brain neural mass modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurolib-dev/neurolib",
    author="Caglar Cakan",
    author_email="cakan@ni.tu-berlin.de",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
