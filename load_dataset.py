import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_preprocess(ds):
    for m in range(len(ds)):

        main_words = re.sub('[^a-zA-Z]', ' ', str(ds[m]))                                      # Retain only alphabets
        main_words = (main_words.lower()).split()
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]  # Remove stopwords

        lem = WordNetLemmatizer()
        main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1]                 # Group different forms of the same word

        main_words = ' '.join(main_words)
        ds[m] = main_words

    return ds


def load_and_preprocess():
		df = pd.read_csv('data/cyberbullying_tweets.csv')
		df.dropna(inplace=True)
		X = df['tweet_text'].values
		y = df['cyberbullying_type'].values

		X = text_preprocess(X.copy())

		label_encoder = LabelEncoder()
		y = label_encoder.fit_transform(y)

		return X, y
	

def split_data(X, y):
      
		X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=123)
		X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=123)
		return X_train, y_train, X_val, y_val, X_test, y_test

#TF-IDF
def get_tf_idf():
		X, y = load_and_preprocess()
		X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

		td = TfidfVectorizer(max_features=3000)
		X_train = td.fit_transform(X_train).toarray()
		X_val = td.transform(X_val).toarray()
		X_test = td.transform(X_test).toarray()

		return X_train, y_train, X_val, y_val, X_test, y_test

#glove embedding

def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab 

def text_to_embeddings(text_data, tokenizer, embedding_matrix):
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    embeddings = []
    for seq in padded_sequences:
        seq_embedding = [embedding_matrix[word_idx] for word_idx in seq if word_idx != 0]
        if not seq_embedding:
            seq_embedding = [np.zeros(embedding_dim)]  # Handle empty sequences
        embeddings.append(np.mean(seq_embedding, axis=0))

    return np.vstack(embeddings)

def get_embedding_data():
    X, y = load_and_preprocess()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    embedding_dim = 300
    embedding_matrix_vocab = embedding_for_vocab(
    'data/glove.6B.300d.txt', tokenizer.word_index,
  	embedding_dim)
    
    X = text_to_embeddings(X, tokenizer, embedding_matrix_vocab)
    return split_data(X, y)
