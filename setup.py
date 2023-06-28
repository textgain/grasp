import os

from setuptools import setup

setup(
                  name = 'grasp',
               version = '2.7',
           description = 'Simple NLP toolkit',
               license = 'BSD',
                author = 'Textgain',
          author_email = 'info@textgain.com',
                   url = 'https://github.com/textgain/grasp',
               scripts = [],
              packages = ['grasp', 'grasp.kb', 'grasp.lm'],
           package_dir = {'grasp': os.path.dirname(__file__)},
          package_data = {
                    '' : ['*.md', '*.csv', '*.json', '*.js', '*.py'],
            'grasp.kb' : ['*.md', '*.csv', '*.json'],
            'grasp.lm' : ['*.md', '*.csv', '*.json', '*.zip']},
        extras_require = {
              'matrix' : 'numpy', 
                 'svd' : 'numpy', 
                'fsel' : 'scipy'},
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