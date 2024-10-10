from setuptools import setup

setup(
    name='mistral-task2vec',
    version='0.1',
    py_modules=['mistral_module', 'task_similarity', 'utils'],  # Include 'mistral_module.py' explicitly
    install_requires=[
        'seaborn',
        'scipy~=1.14.0',
        'matplotlib',
        'omegaconf',
        'fastcluster',
        'torch~=2.4.0',
        'torchvision',
        'numpy~=1.26.4',
        'pandas',
        'hydra-core',
        'scikit-learn',
        'tqdm~=4.66.4',
        'transformers~=4.45.2',
        'datasets~=3.0.1',
    ],
)
