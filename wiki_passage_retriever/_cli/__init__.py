import os

import click
from .retrieve import retrieve

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """CLI for wiki_passage_retriever."""
    pass

cli.add_command(retrieve)
