from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from typing import List
from numpy import ndarray
import torch
from annoy import AnnoyIndex
from .utils import retrieve_wiki_page
import json
import os


EMBEDDING_DIMENSION = 768


def encode_passages(passages: List[str]) -> ndarray:
    with torch.no_grad():
        tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        input_ids = tokenizer(passages, return_tensors='pt', padding=True)["input_ids"]
        return model(input_ids).pooler_output.numpy()


def encode_question(question: str) -> ndarray:
    with torch.no_grad():
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        input_ids = tokenizer(question, return_tensors='pt')["input_ids"]
        return model(input_ids).pooler_output.numpy()[0]


def annoy_index_passages(passages: List[str], index_dir_path: str):
    # Create index directiory, if not exist:
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)


    embeddings = encode_passages(passages)
    indexer = AnnoyIndex(EMBEDDING_DIMENSION, metric='dot')

    for i, embedding in enumerate(embeddings):
        indexer.add_item(i, embedding)

    indexer.build(10)  # use 10 trees
    indexer.save("{}/index.ann".format(index_dir_path))

    # save the text:
    with open("{}/texts.json".format(index_dir_path), "w") as f:
        json.dump(passages, f)


def index_wikipedia(query: str, index_dir_path: str):
    annoy_index_passages(retrieve_wiki_page(query), index_dir_path)


def retrieve_by_index(index_dir_path: str, question: str, top_k: int=1):
    question_embedding = encode_question(question)
    indexer = AnnoyIndex(EMBEDDING_DIMENSION, 'dot')
    indexer.load("{}/index.ann".format(index_dir_path))
    inds = indexer.get_nns_by_vector(question_embedding, top_k)

    with open("{}/texts.json".format(index_dir_path), "r") as f:
        passages = json.load(f)
        return [passages[ind] for ind in inds]


