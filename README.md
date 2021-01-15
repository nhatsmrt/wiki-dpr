# Wikipedia Passage Retriever

This package allows for retrieving Wikipedia passages (i.e paragraphs) relevant to a question.

Under the hood, it uses a [dense passage retriever](https://arxiv.org/pdf/2004.04906.pdf), with pretrained model from HuggingFace's [transformers](https://github.com/huggingface/transformers) library.

## Usage

To install the package (which comes with the command-line tool), run the following command in terminal:
```
  pip install wiki-passage-retriever
```

The easiest way to play with the package is to use the command line tool. For instance:
```
# Indexing a wikipedia page:
wikiretriever index -q="Nelson Mandela" -f nelsonindex

# Retrieve relevant passages from index:
wikiretriever indexed-retrieve -q="Who was Nelson Mandela?" -f nelsonindex -k 5

# Slow retrieval:
  wikiretriever retrieve --query="Nelson Mandela" --question="Who was Nelson Mandela's father?" --topk=5
```

I also provide a simple [flask application](flask-app/) to retrieve and display the results.

[Colab Notebook Examples](https://colab.research.google.com/drive/1szwoqAAGgwKossSQenCFIvrWoX_CD_QU?usp=sharing)

## TODO:
  * ~~Add option to retrieve k best passages~~
  * (Maybe) retrieve individual sentences instead of paragraphs?
  * (When the bug is fixed) Switch to out-of-the-box Huggingface's tokenizer.
  * Add option to run model on GPU
  * Add option to search from different wikipedia articles (e.g first k results from search query)
  * Add option to control text truncation (for now, always use full text).
  * Extract span (instead of outputting entire text)
