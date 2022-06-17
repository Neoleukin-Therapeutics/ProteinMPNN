import setuptools
from codecs import open
from os import path

dir_path = path.abspath(path.dirname(__file__))

with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='vanilla_proteinmpnn',
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