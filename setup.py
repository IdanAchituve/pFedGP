from io import open
from os import path

from setuptools import setup  # find_packages

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

######
# Export requirements file with: pip list --format=freeze > requirements.txt
# Might need to install with: pip install -U pip
######
# get reqs
def requirements():
    list_requirements = []
    with open('requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
      name='pFedGP',
      version='1.0.0',  # Required
      description='',  # Optional
      long_description='',  # Optional
      long_description_content_type='text/markdown',  # Optional (see note above)
      url='',  # Optional
      author='',  # Optional
      author_email='',  # Optional
      packages=['pFedGP'],
      python_requires='>=3.7',
      install_requires=requirements()  # Optional
)
