# Wikipedia Passage Retriever

This package allows for retrieve a wikipedia passage (i.e paragraph) that is relevant to a question.

Under the hood, it uses a [dense passage retriever](https://arxiv.org/pdf/2004.04906.pdf), with pretrained model from HuggingFace's [transformers](https://github.com/huggingface/transformers) library.

To install the package (which comes with the command-line tool):
```
  pip install wiki-passage-retriever
```

To use the CLI:
```
  wikiretriever retrieve --query="Nelson Mandela" --question="Who is Nelson Mandela?"
```

## TODO:
  * Add option to retrieve k best passages
  * (Maybe) retrieve individual sentences instead of paragraphs?
  * (When the bug is fixed) Switch to out-of-the-box Huggingface's tokenizer.
  * Add option to run model on GPU
  * Add option to search from different wikipedia articles (e.g first k results from search query)
  * Add option to control text truncation
