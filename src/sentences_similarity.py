
import pandas as pd
import spacy
import en_core_web_md

from constants import (DATASET_PATH, TEST_SENTENCE, 
                       MAX_SIMILAR_SENTENCES_TO_SHOW)

def show_sentences_similarity():
    """ 
    Show some similar and non similar sentences to a test sentence.
    """

    # Load the dataset which contains the sentences in pandas DataFrame
    source_df = pd.read_csv(DATASET_PATH, header=None, names=["text"])
    # Get the sentences in a list
    sentences = list(source_df.text)

    # Load the statistical model of spaCy which contains the word embeddings
    print('Loading the spaCy statistical model...')
    nlp = en_core_web_md.load()

    # Create a Doc from the text of a test sentence
    example_document = nlp.make_doc(TEST_SENTENCE)
    print(f'\nTest sentence: "{TEST_SENTENCE}"')

    # Find some similar and non similar sentences to the test sentence
    print('Finding some similar and non similar sentences...')
    similar_sentences = []
    non_similar_sentences = []
    index = 0
    # Define a bool variable to control when stop of finding sentences
    stop_finding_sentences = False
    while not stop_finding_sentences:

        # Create a Doc from the text of the sentence
        document = nlp.make_doc(sentences[index])

        # Find large enough sentences
        if len(document.text) > 7:

            # Check if doc is valid, including having a valid word vector
            if document and document.vector_norm:

                # Compute similarity between the sentences
                similarity = example_document.similarity(document)
            
                if similarity > 0.9:
                    similar_sentences.append(document.text)
                if similarity < 0.1:
                    non_similar_sentences.append(document.text)

        # Stop find sentences when we have at least three similar sentences 
        # and three non similar sentences. Or if there are no more sentences
        enough_similar_sentences = (
            len(similar_sentences) >= MAX_SIMILAR_SENTENCES_TO_SHOW
        )
        enough_non_similar_sentences = (
            len(non_similar_sentences) >= MAX_SIMILAR_SENTENCES_TO_SHOW
        )
        there_are_more_sentences = index < (len(sentences)-1)

        stop_finding_sentences = (
            enough_similar_sentences and
            enough_non_similar_sentences or
            not there_are_more_sentences
        )
        index += 1
        
    # Show no more than N similar and non simlar sentences to the test sentence
    print('\nSome similar sentences:')
    [print(similar_sentence) for similar_sentence 
        in similar_sentences[:MAX_SIMILAR_SENTENCES_TO_SHOW]]
    print('\nSome non similar sentences:')
    [print(non_similar_sentence) for non_similar_sentence 
        in non_similar_sentences[:MAX_SIMILAR_SENTENCES_TO_SHOW]]

if __name__ == "__main__":
    show_sentences_similarity()
