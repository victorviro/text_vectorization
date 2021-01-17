
from tqdm import tqdm
import pandas as pd
import en_core_web_md

from constants import DATASET_PATH, SENTENCES_EMBEDDINGS_PATH


def vectorize_reviews():
    """ 
    Load the dataset of reviews, generate the sentence embeddings
    and save them.
    """

    # Load the dataset which contains the sentences in pandas DataFrame
    source_df = pd.read_csv(DATASET_PATH, header=None, names=["text"])
    # Get the sentences in a list
    sentences = list(source_df.text)
    print(f'Number of sentences in the dataset: {len(sentences)}')

    # Load the statistical model of spaCy which contains the word embeddings
    print('Loading the spaCy statistical model...')
    nlp = en_core_web_md.load()

    # Get the embeddings of the sentences
    print('Getting embeddings of the sentences')
    sentence_embeddings = vectorize_sentences(nlp, sentences)
    print('Obtained embeddings of the sentences')
    
    # Convert the embeddings of the sentences to a DataFrame
    sentence_embeddings_df = pd.DataFrame(sentence_embeddings)
    print(f'Lenght of the embeddings: {sentence_embeddings_df.shape[1]}')
    # Save the embeddings of the sentences
    sentence_embeddings_df.to_csv(SENTENCES_EMBEDDINGS_PATH, index=False)
    print(f'Embeddings of the sentences saved in {SENTENCES_EMBEDDINGS_PATH}')    


def vectorize_sentences(nlp, sentences):
    """
    Get the embeddings/vectors of the sentences. 
    """

    sentence_embeddings = []
    for sentence in tqdm(sentences):
        # Create a Doc from the text of the sentence
        document = nlp.make_doc(sentence)
        # Get the embedding of the sentence (average of its token vectors)
        sentence_embedding = document.vector
        sentence_embeddings.append(sentence_embedding)

    return sentence_embeddings

if __name__ == "__main__":
    vectorize_reviews()
