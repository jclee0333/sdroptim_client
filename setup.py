# coding: utf-8
'''Setup sdroptim package.'''
from codecs import open
from setuptools import setup, find_packages

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name             = 'sdroptim_client',
    version          = '0.1.0',
    packages         = find_packages(),
    description      = 'Hyperparameter Optimization for KISTI Science Data Repository (for client)',
    long_description = readme,
    long_description_content_type = 'text/x-rst',
    license          = '',
    author           = 'Jeongcheol Lee',
    author_email     = 'jclee@kisti.re.kr',
    url              = 'https://github.com/jclee0333/sdroptim_client',
    #download_url     = 'Git에 저장된 whl 배포용 압축파일',
    #packages        =   ['sdroptim_client'],
    package_data    =   {'sdroptim_client':['./sdroptim_client/conf/run_conf.json']},
    include_package_data = True,
    install_requires = ['optuna==2.0.0',
                        'psycopg2-binary',
                        'easydict',
                        'astunparse',
                        'numpy',
                        'plotly',
                        'requests',
                        'pandas',],
    classifiers      = ['Programming Language :: Python :: 3.6',
                        'Intended Audience :: Korea Institute of Science and Technology Information',
                        'License :: MIT License']
    )

