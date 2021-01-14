from setuptools import setup

with open('requirements.txt', 'r') as req_file:
    requirements = [line[:-1] for line in req_file if len(line) > 1]

setup(
    name='wiki_passage_retriever',
    version='0.1.0',
    author="Nhat Pham",
    author_email="nphamcs@gmail.com",
    description="A small tool for retrieving relevant passages to a question from Wikipedia",
    py_modules=['wiki_passage_retriever'],
    install_requires=requirements,
    url="https://github.com/nhatsmrt/wiki-dpr",
    entry_points={
        'console_scripts': [
            "wikiretriever = wiki_passage_retriever._cli:cli",
        ],
    }
)
