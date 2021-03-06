from transformers import DPRReader
from transformers.models.dpr import DPRReaderOutput
from .dpr_tokenizer import MyDPRReaderTokenizer
from .utils import retrieve_wiki_page
from typing import List, Union, Tuple, Dict
import torch
from numpy import ndarray


__all__ = ['get_most_relevant_spans_from_wiki', 'get_most_relevant_passages']


def process_with_dpr_reader(passages: List[str], titles: Union[List[str], str], question: str) -> Tuple[MyDPRReaderTokenizer, Dict[str, List[List[int]]], DPRReaderOutput]:
    """
    Processes passages (along with the titles and question) using a dense passage retrieval (DPR) model.

    :param passages:
    :param titles:
    :param question:
    :return: The tokenizer, encoded inputs, and the outputs of the DPR model.
    """
    if isinstance(titles, str):
        return process_with_dpr_reader(passages, [titles] * len(passages), question)

    with torch.no_grad():
        tokenizer = MyDPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
        encoded_inputs = tokenizer(
            questions=question,
            titles=titles,
            texts=passages,
            return_tensors='pt',
            truncation=False,
            padding=True
        )
        return tokenizer, encoded_inputs, model(**encoded_inputs)


def get_relevance_scores(passages: List[str], titles: Union[List[str], str], question: str) -> ndarray:
    """
    Computes the relevance score of the passages with respect to a question.

    :param passages:
    :param titles: a list of corresponding titles (or a single title if all passages are in the same article),
    :param question:
    :return: the relevance score of the passages with respect to the question.
    """
    _, _, outputs = process_with_dpr_reader(passages, titles, question)
    return outputs.relevance_logits.numpy()


def get_most_relevant_spans(passages: List[str], titles: Union[List[str], str], question: str, top_k: int=1) -> List[str]:
    """
    Gets the most relevant spans from the list of passages

    :param passages:
    :param titles:
    :param question:
    :param top_k: number of relevant spans to retrieve.
    :return: list of the most relevant spans.
    """
    tokenizer, encoded_inputs, outputs = process_with_dpr_reader(passages, titles, question)
    return [span.text for span in tokenizer.decode_best_spans(encoded_inputs, outputs, num_spans=top_k)]


def get_most_relevant_spans_from_wiki(search_query: str, question: str, top_k: int=1) -> List[str]:
    """
    Gets the most relevant spans to a question from all the passages of a Wikipedia page.

    :param search_query: the search query for Wikipedia. Will return the passages from the first result.
    :param question:
    :param top_k: number of relevant spans to retrieve.
    :return: list of the most relevant spans.
    """
    return get_most_relevant_spans(retrieve_wiki_page(search_query), search_query, question, top_k)


def get_most_relevant_passages(search_query: str, question: str, top_k: int=1) -> List[str]:
    """
    Retrieves the most relevant passages to a question from a Wikipedia page

    :param search_query: the search query for Wikipedia. Will return the passages from the first result.
    :param question:
    :param top_k: number of relevant passages to retrieve.
    :return: list of the most relevant passages.
    """
    passages = retrieve_wiki_page(search_query)
    relevance_scores = get_relevance_scores(passages, search_query, question)

    # get top_k relevance scores
    # (based on https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array)
    top_k_ind = relevance_scores.argsort()[-top_k:][::-1]
    return [passages[ind] for ind in top_k_ind]

