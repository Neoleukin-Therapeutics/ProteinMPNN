import setuptools
from os import environ
from dunamai import Version

git_version = Version.from_git().serialize(metadata=False)
VERSION = environ['VERSION'] if 'VERSION' in environ else git_version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vanilla_proteinmpnn',
    version=VERSION,
    packages=setuptools.find_packages(),
    license='MIT',
    description='ProteinMPNN',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='@dauparas',
    use_scm_version={'local_scheme': 'no-local-version'},

    setup_requires=['setuptools_scm'],
    install_requires=['numpy', 'torch'],
    scripts=[
        'vanilla_proteinmpnn/protein_mpnn_run.py',
    ],
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.x',
    ],
)