#!/usr/bin/env python3

'''
ChartPeer-SDK python install setup.
Package information are drawn from package.json.
'''

from os import system, name
from json import load
from setuptools import setup, find_packages
import pip

# check for OS dep. python command
pip_command = 'pip3'
if name == 'nt':
    pip_command = 'python -m pip'

# load package information for build
with open('package.json') as f:
    body = load(f)

# upgrade pip
system(f'{pip_command} install --upgrade pip')


# declare all dependencies
pip_dependencies = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'sklearn'
]

# run the setup
setup( name = body['name'],
    version = body['version'],
    packages = find_packages(),
    author = body['author'],
    author_email = body['author_email'],
    description = body['description'],
    url = body['url'],
    install_requires = pip_dependencies
)

# upgrade newly installed science modules
for module in ['numpy', 'scipy']:
    system(f'{pip_command} install --upgrade {module}')