import numpy as np
from scipy.stats import spearmanr
from nltk.corpus import brown, wordnet as wn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
import nltk


nltk.download('brown')
nltk.download('wordnet')


def load_simlex999(file_path):
    word_pairs = []
    true_scores = []
    pos_tags = []
    concreteness_diff = []

    with open(file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            word_pairs.append((parts[0], parts[1]))
            true_scores.append(float(parts[3]))
            pos_tags.append(parts[2])
            concreteness_diff.append(abs(float(parts[4]) - float(parts[5])))
    return word_pairs, true_scores, pos_tags, concreteness_diff


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def prepare_features(word_pairs, embeddings, pos_tags, concreteness_diff):
    features = []
    for (word1, word2), pos, conc_diff in zip(word_pairs, pos_tags, concreteness_diff):
        vec1 = embeddings.get(word1, np.zeros(300))
        vec2 = embeddings.get(word2, np.zeros(300))
        similarity = cosine_similarity(vec1, vec2)

        
        word1_set = set(wn.synsets(word1))
        word2_set = set(wn.synsets(word2))
        if len(word1_set.union(word2_set)) > 0:
            jaccard_similarity = len(word1_set.intersection(word2_set)) / len(word1_set.union(word2_set))
        else:
            jaccard_similarity = 0.0

        wordnet_similarity = extract_wordnet_features(word1, word2)
        pos_match = 1 if pos in {'n', 'v', 'a'} else 0

        features.append([similarity, jaccard_similarity, wordnet_similarity, pos_match, conc_diff])

    return np.array(features)


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    correlation, _ = spearmanr(y_test, predictions)
    print(f"Spearman correlation: {correlation}")
    return correlation


def train_nn(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    predictions = model.predict(X_test).flatten()
    correlation, _ = spearmanr(y_test, predictions)
    print(f"Spearman correlation (NN): {correlation}")
    return correlation

def main():
    print("Loading SimLex-999 dataset...")
    word_pairs, true_scores, pos_tags, concreteness_diff = load_simlex999('SimLex-999.txt')

    print("Loading GloVe embeddings...")
    embeddings = load_glove_embeddings('glove.6B.300d.txt')

    print("Preparing features...")
    X = prepare_features(word_pairs, embeddings, pos_tags, concreteness_diff)

    print("Training and evaluating Gradient Boosting model...")
    correlation_gb = train_and_evaluate(X, np.array(true_scores))

    print("Training and evaluating Neural Network model...")
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(true_scores), test_size=0.2, random_state=42)
    correlation_nn = train_nn(X_train, y_train, X_test, y_test)

    # Analyze coverage
    simlex_words = set(word for pair in word_pairs for word in pair)
    covered_words = simlex_words.intersection(embeddings.keys())
    coverage = len(covered_words) / len(simlex_words)
    print(f"SimLex-999 vocabulary coverage: {coverage:.2%}")

if __name__ == "__main__":
    main()
