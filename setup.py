
from setuptools import setup, find_packages

setup(
    name='extended_ccm',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'skccm'
    ],
    description='Extended Convergent Cross Mapping (CCM) based on Sugihara et al.',
    author='Your Name',
    author_email='you@example.com',
    url='https://github.com/your-repo/extended-ccm',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
