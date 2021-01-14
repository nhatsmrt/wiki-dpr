"""
This is HuggingFace transformers' DPRTokenizer, but with a small bug fixed. See:
https://github.com/huggingface/transformers/issues/9555
"""

from typing import Optional, Union
from transformers.tokenization_utils_base import BatchEncoding, TensorType
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.dpr.tokenization_dpr import CustomDPRReaderTokenizerMixin


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
    },
}

READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,
    "facebook/dpr-reader-multiset-base": 512,
}


READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},
}


class MyCustomDPRReaderTokenizerMixin(CustomDPRReaderTokenizerMixin):
    def __call__(
        self,
        questions,
        titles: Optional[str] = None,
        texts: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs
    ) -> BatchEncoding:
        if titles is None and texts is None:
            return super().__call__(
                questions,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        elif titles is None or texts is None:
            text_pair = titles if texts is None else texts
            return super().__call__(
                questions,
                text_pair,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        assert len(titles) == len(
            texts
        ), "There should be as many titles than texts but got {} titles and {} texts.".format(len(titles), len(texts))
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)


class MyDPRReaderTokenizer(MyCustomDPRReaderTokenizerMixin, BertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["attention_mask"]
