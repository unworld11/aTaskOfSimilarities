import numpy as np
from scipy.stats import spearmanr
from nltk.corpus import brown, wordnet as wn
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import nltk
from sklearn.ensemble import GradientBoostingRegressor

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

# Step 2: Prepare Brown corpus
def prepare_brown_corpus(max_tokens=1000000):
    words = brown.words()[:max_tokens]
    return [word.lower() for word in words]

# Step 3: Build vocabulary and word-to-id mapping
def build_vocab(corpus, min_count=5):
    word_counts = Counter(corpus)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    return vocab, word_to_id

# Step 4: Build co-occurrence matrix
def build_cooccurrence_matrix(corpus, word_to_id, window_size=5):
    vocab_size = len(word_to_id)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))

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

# Step 5: Compute PPMI
def compute_ppmi(cooccurrence_matrix, k=1):
    total_count = np.sum(cooccurrence_matrix)
    word_counts = np.sum(cooccurrence_matrix, axis=1)
    context_counts = np.sum(cooccurrence_matrix, axis=0)

    expected = np.outer(word_counts, context_counts) / total_count
    pmi = np.log(np.maximum(cooccurrence_matrix * total_count / expected, 1e-8))
    ppmi = np.maximum(pmi - np.log(k), 0)

    return ppmi

# Step 6: Create word vectors using truncated SVD
def create_word_vectors(ppmi_matrix, dim=300):
    svd = TruncatedSVD(n_components=dim, random_state=42)
    word_vectors = svd.fit_transform(ppmi_matrix)
    return word_vectors

# Step 7: Compute cosine similarity
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # Return 0 for zero-length vectors

    return np.dot(vec1, vec2) / (norm1 * norm2)

# Step 8: Extract WordNet-based features
def extract_wordnet_features(word1, word2):
    synonyms1 = set(wn.synsets(word1))
    synonyms2 = set(wn.synsets(word2))

    if synonyms1 and synonyms2:
        # Find the maximum path similarity between any synset pairs
        max_similarity = max((s1.path_similarity(s2) for s1 in synonyms1 for s2 in synonyms2 if s1.path_similarity(s2)), default=0)
    else:
        max_similarity = 0
    return max_similarity

# Step 9: Prepare feature matrix
def prepare_features(word_pairs, word_vectors, word_to_id, pos_tags, concreteness_diff):
    features = []
    for (word1, word2), pos, conc_diff in zip(word_pairs, pos_tags, concreteness_diff):
        if word1 in word_to_id and word2 in word_to_id:
            vec1 = word_vectors[word_to_id[word1]]
            vec2 = word_vectors[word_to_id[word2]]
            similarity = cosine_similarity(vec1, vec2)

            # Calculate Jaccard similarity based on word sets
            word1_set = set(wn.synsets(word1))
            word2_set = set(wn.synsets(word2))
            if len(word1_set.union(word2_set)) > 0:
                jaccard_similarity = len(word1_set.intersection(word2_set)) / len(word1_set.union(word2_set))
            else:
                jaccard_similarity = 0.0
        else:
            similarity = 0.0  # Default score for OOV words
            jaccard_similarity = 0.0

        wordnet_similarity = extract_wordnet_features(word1, word2)
        pos_match = 1 if pos in {'n', 'v', 'a'} else 0

        features.append([similarity, jaccard_similarity, wordnet_similarity, pos_match, conc_diff])

    return np.array(features)

# Step 10: Train and evaluate the model
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # Adjust alpha for regularization strength
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    correlation, _ = spearmanr(y_test, predictions)
    print(f"Spearman correlation: {correlation}")
    return correlation

# Main execution
def main():
    print("Loading SimLex-999 dataset...")
    word_pairs, true_scores, pos_tags, concreteness_diff = load_simlex999('SimLex-999.txt')

    print("Preparing Brown corpus...")
    corpus = prepare_brown_corpus()

    print("Building vocabulary...")
    vocab, word_to_id = build_vocab(corpus)

    print("Building co-occurrence matrix...")
    cooccurrence_matrix = build_cooccurrence_matrix(corpus, word_to_id)

    print("Computing PPMI...")
    ppmi_matrix = compute_ppmi(cooccurrence_matrix)

    print("Creating word vectors...")
    word_vectors = create_word_vectors(ppmi_matrix)

    print("Preparing features...")
    X = prepare_features(word_pairs, word_vectors, word_to_id, pos_tags, concreteness_diff)

    print("Training and evaluating the model...")
    correlation = train_and_evaluate(X, np.array(true_scores))

    # Analyze coverage
    simlex_words = set(word for pair in word_pairs for word in pair)
    covered_words = simlex_words.intersection(vocab)
    coverage = len(covered_words) / len(simlex_words)
    print(f"SimLex-999 vocabulary coverage: {coverage:.2%}")

if __name__ == "__main__":
    main()
