import setuptools

setuptools.setup(
    name='vlad',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='',
    license='MIT',
    author='Tim Hilt',
    author_email='timhilt@live.de',
    description='Implementation of Vector of Locally Aggregated Descriptors (VLAD) Proposed by Jégou et al',
    install_requires=[
        "numpy",
        "scikit-learn",
        "progressbar2"
    ]
)
