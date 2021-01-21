import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from typing import List, Callable


__all__ = ['retrieve_wiki_page']


def remove_citation(paragraph: str) -> str:
    """Remove all citations (numbers in side square brackets) in paragraph"""
    return re.sub(r'\[\d+\]', '', paragraph)


def remove_new_line(paragraph: str) -> str:
    return paragraph.replace("\n", "")


def compose_fns(functions: List[Callable]) -> Callable:
    def ret(input):
        for fn in functions:
            input = fn(input)

        return input

    return ret


def retrieve_page_content(url: str) -> List[str]:
    """
    Retrieve all the page's text paragraphs.

    :param url: URL of the page.
    :return: list of paragraphs of the page.
    """
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    preprocess_fn = compose_fns([remove_citation, remove_new_line])
    paragraphs = list(map(lambda p: preprocess_fn(p.getText()), soup.find_all('p')))

    return paragraphs


def retrieve_wiki_page(query: str) -> List[str]:
    """
    Searches wikpedia for a query, gets the first result, then extracts all the paragraphs.

    :param query: Wikipedia search query.
    :return: list of paragraphs from the first result.
    """
    session = requests.Session()
    api_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query
    }

    response = session.get(url=api_url, params=params)
    data = response.json()

    # only get the first result
    page_id = data['query']['search'][0]['pageid']
    return retrieve_page_content("https://en.wikipedia.org/?curid={}".format(page_id))
