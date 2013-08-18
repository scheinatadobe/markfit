from setuptools import setup, find_packages

setup(
    name='markfit',
    version='0.1.0',
    author='Andrew I. Schein',
    author_email='aschein@adobe.com',
    packages=find_packages(),
    scripts=[],
    url='https://git.corp.adobe.com/aschein/linear',
    license='LICENSE.txt',
    description='High level linear regression library',
    long_description=open('README.txt').read(),
    #install_requires=[
    #    "patsy >= 0.1.0",
    #    "pandas >= 0.10.0",
    #],
)

