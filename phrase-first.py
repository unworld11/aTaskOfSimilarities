import numpy as np
from scipy.stats import spearmanr
from nltk.corpus import brown, wordnet as wn
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import nltk
import logging
import joblib
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('brown', quiet=True)
nltk.download('wordnet', quiet=True)

def load_simlex999(file_path):
    word_pairs, true_scores, pos_tags, concreteness_diff = [], [], [], []

    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                word_pairs.append((parts[0], parts[1]))
                true_scores.append(float(parts[3]))
                pos_tags.append(parts[2])
                concreteness_diff.append(abs(float(parts[4]) - float(parts[5])))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading SimLex-999 dataset: {str(e)}")
        raise

    return word_pairs, true_scores, pos_tags, concreteness_diff

def prepare_brown_corpus(max_tokens=1000000):
    return [word.lower() for word in brown.words()[:max_tokens]]

def build_vocab(corpus, min_count=5):
    word_counts = Counter(corpus)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    return vocab, word_to_id

def build_cooccurrence_matrix(corpus, word_to_id, window_size=5):
    vocab_size = len(word_to_id)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for i, word in enumerate(corpus):
        if word in word_to_id:
            target_id = word_to_id[word]
            start = max(0, i - window_size)
            end = min(len(corpus), i + window_size + 1)

            for j in range(start, end):
                if i != j and corpus[j] in word_to_id:
                    context_id = word_to_id[corpus[j]]
                    cooccurrence_matrix[target_id, context_id] += 1

    return cooccurrence_matrix

def load_glove_embeddings(file_path):
    word_to_vec = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                word_to_vec[word] = vector
    except FileNotFoundError:
        print(f"GloVe embeddings file not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading GloVe embeddings: {str(e)}")
        raise
    return word_to_vec

def extract_external_knowledge_features(word1, word2):
    word1_synsets = wn.synsets(word1)
    word2_synsets = wn.synsets(word2)

    num_synsets_word1 = len(word1_synsets)
    num_synsets_word2 = len(word2_synsets)

    max_similarity = max(
        (s1.wup_similarity(s2) for s1 in word1_synsets for s2 in word2_synsets if s1.wup_similarity(s2) is not None),
        default=0
    )

    return [num_synsets_word1, num_synsets_word2, max_similarity]

def prepare_features(word_pairs, word_vectors, word_to_id, pos_tags, concreteness_diff):
    features = []
    for (word1, word2), pos, conc_diff in zip(word_pairs, pos_tags, concreteness_diff):
        vec1 = word_vectors.get(word1, np.zeros(300))
        vec2 = word_vectors.get(word2, np.zeros(300))
        similarity = cosine_similarity(vec1, vec2)

        word1_set = set(wn.synsets(word1))
        word2_set = set(wn.synsets(word2))
        jaccard_similarity = len(word1_set.intersection(word2_set)) / len(word1_set.union(word2_set)) if word1_set or word2_set else 0.0

        wordnet_features = extract_external_knowledge_features(word1, word2)
        pos_match = 1 if pos in {'n', 'v', 'a'} else 0

        features.append([similarity, jaccard_similarity, *wordnet_features, pos_match, conc_diff])

    return np.array(features)

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 and norm2 else 0.0

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    correlation, _ = spearmanr(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test Spearman correlation: {correlation:.4f}")

    # Save the model
    save_model(best_model, 'gradient_boosting_model.pkl')

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def prepare_phrase_features(phrases, word_vectors):
    features = []
    for phrase1, phrase2 in phrases:
        vec1 = np.mean([word_vectors.get(word, np.zeros(300)) for word in phrase1.split()], axis=0)
        vec2 = np.mean([word_vectors.get(word, np.zeros(300)) for word in phrase2.split()], axis=0)
        similarity = cosine_similarity(vec1, vec2)

        features.append([similarity])

    return np.array(features)

def main():
    # Option to load SimLex-999 dataset and train model
    print("Loading SimLex-999 dataset...")
    word_pairs, true_scores, pos_tags, concreteness_diff = load_simlex999('SimLex-999.txt')

    print("Preparing Brown corpus...")
    corpus = prepare_brown_corpus()

    print("Building vocabulary...")
    vocab, word_to_id = build_vocab(corpus)

    print("Building co-occurrence matrix...")
    cooccurrence_matrix = build_cooccurrence_matrix(corpus, word_to_id)

    print("Loading GloVe embeddings...")
    word_vectors = load_glove_embeddings('glove.6B.300d.txt')

    print("Preparing features for SimLex-999...")
    features = prepare_features(word_pairs, word_vectors, word_to_id, pos_tags, concreteness_diff)

    print("Training and evaluating models...")
    train_and_evaluate(features, true_scores)

    # Save the trained model for SimLex-999
    best_model = load_model('gradient_boosting_model.pkl')

    # Load the PiC Phrase Similarity dataset
    print("Loading PiC Phrase Similarity dataset...")
    ds = load_dataset("PiC/phrase_similarity")
    phrases = [(entry['phrase1'], entry['phrase2']) for entry in ds['train']]
    true_scores = [entry['label'] for entry in ds['train']]

    print("Preparing phrase features...")
    phrase_features = prepare_phrase_features(phrases, word_vectors)

    print("Training and evaluating models for phrase similarity...")
    train_and_evaluate(phrase_features, true_scores)

    # Save model for future use
    best_phrase_model = load_model('gradient_boosting_model.pkl')
    save_model(best_phrase_model, 'phrase_similarity_model.pkl')

if __name__ == "__main__":
    main()
