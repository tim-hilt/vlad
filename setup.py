import setuptools

with open ("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='vlad',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/tim-hilt/vlad',
    license='MIT',
    author='Tim Hilt',
    author_email='timhilt@live.de',
    description='Implementation of Vector of Locally Aggregated Descriptors (VLAD) Proposed by Jégou et al',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scikit-learn",
        "progressbar2"
    ]
)
