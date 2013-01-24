from distutils.core import setup

setup(
    name='linear_model',
    version='0.1.0',
    author='Andrew I. Schein',
    author_email='aschein@adobe.com',
    packages=['linear_model'],
    scripts=[],
    url='https://git.corp.adobe.com/aschein/linear',
    license='LICENSE.txt',
    description='High level linear regression library',
    long_description=open('README.txt').read(),
    install_requires=[
        "patsy >= 0.1.0",
        "pandas >= 0.10.1",
    ],
)

