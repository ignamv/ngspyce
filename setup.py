from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='ngspyce',
      version=version,
      description="Python interface to ngspice",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='ngspice spice geda CAD simulation',
      author='Ignacio Martinez V.',
      author_email='ignamv@gmail.com',
      url='http://github.com/ignamv/ngspyce',
      license='GPLv3',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'numpy'
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
