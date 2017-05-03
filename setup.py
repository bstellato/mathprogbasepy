from setuptools import setup


setup(name='mathprogbasepy',
      version='0.1.0',
      author='Bartolomeo Stellato',
      description='Low level interface for mathematical programming solvers.',
      url='http://github.com/bstellato/mathprogbasepy/',
      package_dir={'mathprogbasepy': 'mathprogbasepy'},
      install_requires=["numpy >= 1.7",
                        "scipy >= 0.13.2"],
      license='Apache 2.0',
      packages=['mathprogbasepy',
                'mathprogbasepy.quadprog',
                'mathprogbasepy.quadprog.solvers',
                'mathprogbasepy.tests'])
