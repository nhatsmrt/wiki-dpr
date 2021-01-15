import click
from ..index import index_wikipedia


@click.command()
@click.option('--query', '-q', help="Search query for wikipedia.")
@click.option('--index-dir-path', '-f', type=click.Path(dir_okay=True, file_okay=False), help="Path to save index.")
def index(query, index_dir_path):
    """Index the first result of the given Wikipedia query for future passage retrieval."""
    if not (query and index_dir_path):
        raise click.BadParameter("\"query\" and \"index_dir_path\" must be provided. For more information, use --help option.")

    index_wikipedia(query, index_dir_path)