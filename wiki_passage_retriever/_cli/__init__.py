import os

import click
from .retrieve import retrieve
from .index import index
from .indexed_retrieve import indexed_retrieve

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """CLI for wiki_passage_retriever."""
    pass

cli.add_command(retrieve)
cli.add_command(index)
cli.add_command(indexed_retrieve)
