from setuptools import setup

setup(
   name='kak_tools',
   version='0.1',
   description='KAK decomposition tools',
   author='People',
   author_email='emails',
   packages=['kak_tools'],  #same as name
   # paulie classifies the DLA that kak_tools.paulie_bridge decomposes; it needs
   # Python >= 3.12.
   install_requires=['pennylane', 'paulie>=0.0.2'],
)
