import numpy as np
import joblib
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load GloVe embeddings
def load_glove_embeddings(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            word_vectors[word] = vector
    return word_vectors

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 and norm2 else 0.0

# Function to compute Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

# Function to prepare features for a phrase pair
def prepare_features_for_phrase_pair(phrase_pair, word_vectors, tfidf_vectorizer):
    phrase1, phrase2 = phrase_pair
    doc1 = nlp(phrase1)
    doc2 = nlp(phrase2)

    # Word embeddings cosine similarity
    vec1 = np.mean([word_vectors.get(word.text, np.zeros(300)) for word in doc1], axis=0)
    vec2 = np.mean([word_vectors.get(word.text, np.zeros(300)) for word in doc2], axis=0)
    cosine_sim = cosine_similarity(vec1, vec2)

    # TF-IDF cosine similarity
    tfidf_matrix = tfidf_vectorizer.transform([phrase1, phrase2])
    tfidf_cosine_sim = cosine_similarity(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1])

    # Jaccard similarity
    jaccard_sim = jaccard_similarity(set(phrase1.split()), set(phrase2.split()))

    # Length difference
    len_diff = abs(len(phrase1) - len(phrase2))

    # Syntactic features (PoS tag similarity)
    pos_tag_similarity = cosine_similarity(
        np.mean([word.vector for word in doc1], axis=0),
        np.mean([word.vector for word in doc2], axis=0)
    )

    # Combine all features into a single feature vector
    features = [cosine_sim, tfidf_cosine_sim, jaccard_sim, len_diff, pos_tag_similarity]
    return np.array(features).reshape(1, -1)

# Function to load and preprocess the PiC dataset from Hugging Face
def load_and_preprocess_pic_dataset(word_vectors, tfidf_vectorizer):
    dataset = load_dataset("PiC/phrase_similarity", split="train")
    phrase_pairs = list(zip(dataset['phrase1'], dataset['phrase2']))
    true_labels = dataset['label']

    features = []
    for phrase_pair in phrase_pairs:
        features.append(prepare_features_for_phrase_pair(phrase_pair, word_vectors, tfidf_vectorizer))

    features = np.vstack(features)
    return features, true_labels

# Function to train and save the Gradient Boosting model
def train_and_save_model(features, labels, model_filename):
    # Initialize the model
    model = GradientBoostingRegressor()

    # Train the model
    model.fit(features, labels)

    # Save the model to a file
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

# Example usage
def main():
    # Load word vectors (e.g., GloVe)
    word_vectors = load_glove_embeddings('glove.6B.300d.txt')

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit the vectorizer on the PiC dataset
    pic_dataset = load_dataset("PiC/phrase_similarity", split="train")
    tfidf_vectorizer.fit(pic_dataset['phrase1'] + pic_dataset['phrase2'])

    # Load and preprocess the PiC dataset
    pic_features, pic_true_labels = load_and_preprocess_pic_dataset(word_vectors, tfidf_vectorizer)

    # Train and save the model
    train_and_save_model(pic_features, pic_true_labels, 'phrase_similarity_model.pkl')

    # Load the saved model
    phrase_model = joblib.load('phrase_similarity_model.pkl')

    # Predict similarity using the model
    pic_predictions = phrase_model.predict(pic_features)

    # Evaluate the model performance
    mse = mean_squared_error(pic_true_labels, pic_predictions)
    spearman_corr, _ = spearmanr(pic_true_labels, pic_predictions)

    print(f"PiC Dataset Test MSE: {mse:.4f}")
    print(f"PiC Dataset Spearman correlation: {spearman_corr:.4f}")

if __name__ == "__main__":
    main()
