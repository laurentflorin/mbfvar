from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Multifrequency Bayesian VAR Model'
LONG_DESCRIPTION = 'Multifrequency Bayesian VAR Model'

# Setting up
setup(
        # the name must match the folder name 'verysimplemodule'
        name="MUFBVAR", 
        version=VERSION,
        author="Laurent Florin",
        author_email="<laurent.florin@efv.admin.ch>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy", "math", "collections", "scipy", "pandas", "datetime", "itertools", "matplotlib", "tqdm", "plotly", "fanchart", "pickle"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Linux",
            "Operating System :: Microsoft :: Windows",
        ]
)