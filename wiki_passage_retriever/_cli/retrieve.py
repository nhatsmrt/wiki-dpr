import click
from ..retrieve import get_most_relevant_passages


@click.command()
@click.option('--query', help="Search query for wikipedia.")
@click.option('--question', help="Question to ask.")
def retrieve(query, question):
    """Retrieve most relevant passage to the question from the wiki page."""
    if not (query and question):
        raise click.BadParameter("\"query\" and \"question\" must be provided. For more information, use --help option.")
    click.echo(get_most_relevant_passages(query, question))
