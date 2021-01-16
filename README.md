# Vectorization of reviews with sentence embeddings

In this repository, we create a sentence embedding for each sentence of the dataset. We use spaCy, in particular, the pretrained statistical model [en_core_web_md](https://spacy.io/models/en#en_core_web_md) which contains 20k unique 300-dim word vectors, and it was trained using the GloVe algorithm on the Common Crawl dataset. To compute the embeddings of the sentences or pieces of sentences, spaCy, by default, will take an average of their token vectors. Note that this type of vectorization does not take into account the order of the words in the sentences (for that we can use contextualized word embeddings with models like ELMo or BERT) (a review of different text vectorizations techniques can be seen [here]()). Finally, we show the capacity of the vectorization to estimate the similarity between sentences.

Download the dataset.
```
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/vectorization/corpus.csv -P data/raw
```

Create virtual environment via venv or conda e install requirements.
```
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Vectorize the sentences (download the spacy model previously). The embeddings of the sentences will store in the directory `data/processed`.
```
python -m spacy download en_core_web_md
python src/vectorize.py
```

Show similarity between sentences. We show some similar and non similar sentences to a test sentence.
```
python src/sentences_similarity.py
```
