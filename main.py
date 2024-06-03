import pandas as pd
import nltk
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# File paths
base_path = 'intermediate_outputs'
os.makedirs(base_path, exist_ok=True)

tokenized_sentences_path = os.path.join(base_path, 'tokenized_sentences.csv')
tokenized_words_path = os.path.join(base_path, 'tokenized_words.csv')
stemmed_words_path = os.path.join(base_path, 'stemmed_words.csv')
lemmatized_words_path = os.path.join(base_path, 'lemmatized_words.csv')
filtered_words_path = os.path.join(base_path, 'filtered_words.csv')

# Load the dataset
file_path = 'is_AI_Generated.csv'
is_AI_Generated_data = pd.read_csv(file_path)
print(is_AI_Generated_data.head())
total_score = is_AI_Generated_data['generated'].sum()
print(f"The total sum of all scores is: {total_score}")

# Handle non-string values in the 'text' column
is_AI_Generated_data['text'] = is_AI_Generated_data['text'].fillna('')  # Replace NaN values with empty strings
text_column = 'text'
corpus = ' '.join(str(review) for review in is_AI_Generated_data[text_column])

# Tokenization: Split the text into words and sentences
if os.path.exists(tokenized_sentences_path) and os.path.exists(tokenized_words_path):
    sentences = pd.read_csv(tokenized_sentences_path, header=None)[0].tolist()
    words = pd.read_csv(tokenized_words_path, header=None)[0].tolist()
else:
    sentences = sent_tokenize(corpus)
    words = word_tokenize(corpus)
    pd.DataFrame(sentences).to_csv(tokenized_sentences_path, index=False, header=False)
    pd.DataFrame(words).to_csv(tokenized_words_path, index=False, header=False)

# Stemming: Reduce words to their root form
if os.path.exists(stemmed_words_path):
    stemmed_words = pd.read_csv(stemmed_words_path, header=None)[0].tolist()
else:
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    pd.DataFrame(stemmed_words).to_csv(stemmed_words_path, index=False, header=False)

# Lemmatization: Further reduce the stemmed words by considering their context
if os.path.exists(lemmatized_words_path):
    lemmatized_words = pd.read_csv(lemmatized_words_path, header=None)[0].tolist()
else:
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    pd.DataFrame(lemmatized_words).to_csv(lemmatized_words_path, index=False, header=False)

# StopWordRemoval: Eliminate common words that may not be useful for analysis
if os.path.exists(filtered_words_path):
    filtered_words = pd.read_csv(filtered_words_path, header=None)[0].tolist()
else:
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
    pd.DataFrame(filtered_words).to_csv(filtered_words_path, index=False, header=False)

# Ensure all words are strings
filtered_words = [str(word) for word in filtered_words]

# Join filtered words into a single document
filtered_text = ' '.join(filtered_words)

# Print the results
print("Original Text:")
print(corpus[:500])  # Displaying a portion of the original text
print("\nTokenization:")
print("Number of sentences:", len(sentences))
print("Number of words:", len(words))
print("\nStemming:")
print(stemmed_words[:20])  # Displaying the first 20 stemmed words
print("\nLemmatization:")
print(lemmatized_words[:20])  # Displaying the first 20 lemmatized words
print("\nStopWordRemoval:")
print(filtered_words[:20])  # Displaying the first 20 filtered words

# --------------------------------------------------------------
# Part 03

# Bag-of-Words (BoW)
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform([filtered_text])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([filtered_text])

# n-grams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))  # Example for bi-grams
ngram_matrix = ngram_vectorizer.fit_transform([filtered_text])

# Visualization - Word Clouds
def visualize_word_cloud(matrix, vectorizer):
    word_freq = dict(zip(vectorizer.get_feature_names_out(), matrix.sum(axis=0).tolist()[0]))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Visualize Bag-of-Words
visualize_word_cloud(bow_matrix, bow_vectorizer)

# Visualize TF-IDF
visualize_word_cloud(tfidf_matrix, tfidf_vectorizer)

# Visualize n-grams
visualize_word_cloud(ngram_matrix, ngram_vectorizer)
