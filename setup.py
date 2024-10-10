from setuptools import setup, find_packages


# Helper function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith("#")]


setup(
    name='mistral-task2vec',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)
