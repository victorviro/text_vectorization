# Vectorization of reviews with sentence embeddings

## Description
In this repository, we create a **sentence embedding** for each sentence of the dataset. 

## Details

We use [spaCy](https://spacy.io/), in particular, the pretrained statistical model [en_core_web_md](https://spacy.io/models/en#en_core_web_md) which contains 20k unique 300-dim word vectors, and it was trained using the GloVe algorithm on the Common Crawl dataset. 

To compute the embeddings of the sentences, spaCy, by default, will take an average of their token vectors. Note that this type of vectorization does not take into account the order of the words in the sentences (for that we can use contextualized word embeddings with models like ELMo or BERT) (a review of different text vectorizations techniques can be seen [here]()). 

Finally, we show the capacity of the vectorization to estimate the similarity between sentences.

## Set up
Download the dataset.
```shell
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/vectorization/corpus.csv -P data/raw
```

Create virtual environment via venv or conda e install requirements.
```shell
cd src
# Create virtual env
pip install -r requirements.txt
```

## Vectorization
Vectorize the sentences. The embeddings of the sentences will be stored in the directory `data/processed`.
```shell
python src/vectorize.py
```

## Sentences similarity
Show similarity between sentences. We show some similar and non similar sentences to a test sentence. The test sentence can be modified in the file `constants.py`.
```shell
python src/sentences_similarity.py
```
