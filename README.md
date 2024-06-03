# Gen-AI Bootcamp 24 - Natural Language Processing Course Assignment

## Part 1: Text Collection and Loading

### Objective:
Collect and load a text dataset from a selected domain into a suitable format for processing.

### Tasks:
- Choose a domain of interest (e.g., healthcare, sports, e-commerce).
- Identify a dataset from your chosen domain (e.g., Kaggle, UCI Machine Learning Repository, government open data portals, APIs).
- Write code to load the dataset into a suitable format (CSV, JSON, or plain text files).
- Ensure the dataset is loaded correctly by displaying the first few rows.

## Part 2: Text Preprocessing

### Objective:
Gain hands-on experience with text preprocessing techniques.

### Tasks:
- Choose a text corpus from the NLTK library.
- Perform the following preprocessing steps:
  - Tokenization: Split the text into words and sentences.
  - Stemming: Reduce words to their root form.
  - Lemmatization: Further reduce the stemmed words by considering their context.
  - StopWordRemoval: Eliminate common words that may not be useful for analysis.
- Properly comment each step.
- Discuss the impact of each preprocessing step on the corpus.

## Part 3: Feature Extraction Techniques

### Objective:
Understand and apply text data transformation into machine-readable vectors.

### Tasks:
- Using the pre-processed text from Part 2, implement the following feature extraction methods:
  - Bag-of-words
  - TF-IDF
  - n-grams
- Explain when one method might be preferred over the others and provide reasoning.
- Visualize the most common terms with each method using a word cloud.

## Part 4: Word Embeddings

### Objective:
Explore word embeddings and their applications.

### Tasks:
- Apply Word2Vec, GloVe, and FastText embeddings to a sample text and your dataset using pre-trained models from Gensim’s Data repository.
- Visualize the word embeddings using t-SNE to observe clusters of similar words.

## Part 5: Model Training and Evaluation

### Objective:
Understand RNNs and their ability to handle sequence data.

### Tasks:
- Construct a simple RNN model using TensorFlow and Keras libraries.
- Train LSTM and GRU networks on your problem using the loaded dataset.
- Compare the performance of LSTM and GRU networks on the task.
- Analyze the long-term dependencies captured by each model.

## Part 6: Visualization and Interpretation

### Objective:
Visualize and interpret the results to gain insights from the model's performance.

### Tasks:
- Use visualizations (e.g., word clouds, confusion matrices) to understand the data and model outputs.
- Provide human-readable interpretations of the model's predictions and decisions.
- Discuss the implications of the findings in the context of the selected domain.

## Submission Guidelines:
Submit a detailed report including code, outputs, and your analysis of the results.
