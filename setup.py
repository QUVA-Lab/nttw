from setuptools import setup, find_packages

setup(name='nttw',
      version='0.1',
      description='Code for AISTATS 2023 paper "No time to waste"',
      url='',
      author='Rob Romijnders',
      author_email='romijndersrob@gmail.com',
      license='LICENSE.txt',
      install_requires=[
          'numpy',
          'wandb',
          'scipy',
          'sklearn',
          'matplotlib',
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
