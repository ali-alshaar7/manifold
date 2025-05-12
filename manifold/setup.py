from setuptools import setup, find_packages

setup(
    name='manifold',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A DSL for GPU programming with kernel support',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here, e.g.:
        # 'numpy',
        # 'cupy',  # For GPU support
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)