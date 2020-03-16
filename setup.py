from setuptools import setup

setup(
                  name = 'Grasp',
               version = '2.0',
           description = 'Simple NLP toolkit',
               license = 'BSD',
                author = 'Textgain',
          author_email = 'info@textgain.com',
                   url = 'https://github.com/textgain/grasp',
               scripts = [],
              packages = [],
          package_data = {
                  ''   : ['*.csv', '*.json', '*.js', '*.md'],
                  'kb' : ['*.csv', '*.json'],
                  'lm' : ['*.csv', '*.json']},
        extras_require = {
                 'svd' : 'numpy', 
                'fsel' : 'scipy'},
           classifiers = [
   'Development Status :: 5 - Production/Stable',
          'Environment :: Web Environment',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
     'Natural Language :: Dutch',
     'Natural Language :: English',
     'Operating System :: OS Independent',
 'Programming Language :: JavaScript',
 'Programming Language :: Python',
                'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Software Development :: Libraries :: Python Modules',
                'Topic :: Text Processing :: Linguistic',
                'Topic :: Text Processing :: Markup :: HTML'],
)