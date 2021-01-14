import click
from ..retrieve import get_most_relevant_passages


@click.command()
@click.option('--query', help="Search query for wikipedia.")
@click.option('--question', help="Question to ask.")
@click.option('--topk', type=int, default=1, help="Number of passages to retrieve, in the order of relevance.")
def retrieve(query, question, topk):
    """Retrieve most relevant passage to the question from the wiki page."""
    if not (query and question):
        raise click.BadParameter("\"query\" and \"question\" must be provided. For more information, use --help option.")

    for passage in get_most_relevant_passages(query, question, topk):
        click.echo(passage)
