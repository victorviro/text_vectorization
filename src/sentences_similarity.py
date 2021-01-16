
import pandas as pd
import spacy
import en_core_web_md

from constants import DATASET_PATH, MAX_SIMILAR_SENTENCES_TO_SHOW

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
    example_doc = nlp.make_doc(sentences.pop(11))
    print(f'\nTest sentence: "{example_doc.text}"')

    # Find some similar and non similar sentences to the test sentence
    print('Finding some similar and non similar sentences...')
    similar_sentences = []
    non_similar_sentences = []
    index = 0
    # Define a bool variable to control when stop of finding sentences
    stop_finding_similar_sentences = False
    while not stop_finding_similar_sentences:

        # Create a Doc from the text of the sentence
        doc = nlp.make_doc(sentences[index])

        # Find large enough sentences
        if len(doc.text) > 7:

            # Check if doc is valid, including having a valid word vector
            if doc and doc.vector_norm:

                # Compute similarity between the sentences
                similarity = example_doc.similarity(doc)
            
                if similarity > 0.9:
                    similar_sentences.append(doc.text)
                if similarity < 0.1:
                    non_similar_sentences.append(doc.text)

        # Stop find sentences when we have at least three sentences or
        # there are no more sentences
        stop_finding_similar_sentences = (
            len(similar_sentences) >= MAX_SIMILAR_SENTENCES_TO_SHOW and
            len(non_similar_sentences) >= MAX_SIMILAR_SENTENCES_TO_SHOW or
            index >= (len(sentences)-1)
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
