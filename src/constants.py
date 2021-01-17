import os


PROJECT_PATH = os.getcwd()
DATASET_PATH = f'{PROJECT_PATH}/data/raw/corpus.csv'
EMBEDDINGS_OF_SENTENCES_PATH = (f'{PROJECT_PATH}/data/processed/'
                                'embeddings_of_sentences.csv')

# The test sentence used to find similar sentences
# when showing similarity between sentences
TEST_SENTENCE = 'The restaurant was very small and poor.'
# Maximum number of similar and non similar sentences to show 
MAX_SIMILAR_SENTENCES_TO_SHOW = 3
