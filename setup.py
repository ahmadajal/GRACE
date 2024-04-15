"""Setup."""

from setuptools import find_packages, setup

# Installation
config = {
    'name': 'GNN_recommender_system',
    'version': '1.1.1',
    'description': 'GNN for recommender system project.',
    'author': 'Hedi Razgallah, Ahmad Ajallooeian, Michalis Vlachos',
    'author_email': 'hedi.razgallah@gmail.com',
    'packages': find_packages(),
    'zip_safe': True
}

setup(**config)