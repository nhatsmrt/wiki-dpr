from transformers import DPRReader
from .dpr_tokenizer import MyDPRReaderTokenizer
from .utils import retrieve_wiki_page
from typing import List, Union
import torch


def get_relevance_scores(passages: List[str], titles: Union[List[str], str], question: str):
    """
    Given a list of passages, a list of corresponding titles (or a single title if all passages are in the same article), and a question,
    returns the relevance score of the passages with respect to the question.
    """
    if isinstance(titles, str):
        return get_relevance_scores(passages, [titles] * len(passages), question)

    with torch.no_grad():
        tokenizer = MyDPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
        encoded_inputs = tokenizer(
            questions=question,
            titles=titles,
            texts=passages,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        outputs = model(**encoded_inputs)
        return outputs.relevance_logits.numpy()



def get_most_relevant_passages(search_query: str, question: str) -> str:
    passages = retrieve_wiki_page("Nelson Mandela")
    relevance_scores = get_relevance_scores(passages, search_query, question)
    return passages[relevance_scores.argmax()]

