from setuptools import setup

import os

PATH = os.path.dirname(__file__)

setup(
                  name = 'grasp',
               version = '3.2.3',
           description = 'Simple NLP toolkit',
               license = 'BSD',
                author = 'Textgain',
          author_email = 'info@textgain.com',
                   url = 'https://github.com/textgain/grasp',
               scripts = [],
              packages = ['grasp',       'grasp.kb',       'grasp.lm',       'grasp.etc'       ],
           package_dir = {'grasp': PATH, 'grasp.kb': 'kb', 'grasp.lm': 'lm', 'grasp.etc': 'etc'},
          package_data = {
                    '' : ['*.md', '*.csv', '*.json', '*.js', '*.txt', '*.py'],
            'grasp.kb' : ['*.md', '*.csv', '*.json'],
            'grasp.lm' : ['*.md', '*.csv', '*.json', '*.zip'],
            'grasp.etc': ['pdf*']},
        extras_require = {
                'Bert' : ['torch', 'transformers'],
              'matrix' : ['numpy'], 
                 'svd' : ['numpy'], 
                'fsel' : ['scipy']},
           classifiers = [
   'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
     'Natural Language :: Dutch',
     'Natural Language :: English',
     'Natural Language :: French',
     'Natural Language :: German',
     'Natural Language :: Spanish',
     'Operating System :: OS Independent',
 'Programming Language :: JavaScript',
 'Programming Language :: Python',
                'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
                'Topic :: Internet :: WWW/HTTP :: WSGI :: Server', 
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Scientific/Engineering :: Visualization',
                'Topic :: Software Development :: Libraries :: Python Modules',
                'Topic :: Text Processing :: Linguistic',
                'Topic :: Text Processing :: Markup :: HTML'],
)

# python -m build --wheel