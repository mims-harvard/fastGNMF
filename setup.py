from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='fastGNMF',
    version='0.1.0',
    author='Kathleen Sucipto',
    author_email='sucipto.kathleen@gmail.com',
    description='A python model for graph regularized non-negative matrix factorization using faiss library for its similarity search.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pypa/sampleproject',
    packages=find_packages(),
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)
