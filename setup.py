from distutils.core import setup
import sys
import os
setup(name='frankmocap',
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='gward@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['renderer', 'demo', 'detectors', 'handmocap', 'inference', 'integration', 'mocap_utils', 'bodymocap'],
     )
