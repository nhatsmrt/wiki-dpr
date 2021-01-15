import click
from ..index import retrieve_by_index


@click.command()
@click.option('--index-dir-path', '-f', type=click.Path(dir_okay=True, file_okay=False), help="Path to save index.")
@click.option('--question', '-q', help="Question to ask.")
@click.option('--topk', '-k', type=int, default=1, help="Number of passages to retrieve, in the order of relevance.")
def indexed_retrieve(index_dir_path, question, topk):
    """Retrieve the top k most relevant passages to a question from index."""
    if not (index_dir_path and question):
        raise click.BadParameter("\"index_dir_path\" and \"question\" must be provided. For more information, use --help option.")

    for passage in retrieve_by_index(index_dir_path, question, topk):
        click.echo(passage)