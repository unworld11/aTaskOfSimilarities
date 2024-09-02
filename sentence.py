import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')

! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet # temp fix for lookup error.

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

word_vectors = load_glove_embeddings('/kaggle/input/glove6b300dtxt/glove.6B.300d.txt')
embedding_dim = 300

def create_phrase_embedding(phrase, word_vectors, embedding_dim):
    phrase = preprocess_text(phrase)
    tokens = nltk.word_tokenize(phrase)
    embeddings = [word_vectors.get(word, np.zeros(embedding_dim)) for word in tokens]
    return np.stack(embeddings) if embeddings else np.zeros((1, embedding_dim))

class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels, word_vectors, embedding_dim, max_len=20):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.word_vectors = word_vectors
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2 = self.sentence_pairs[idx]
        label = self.labels[idx]
        
        emb1 = create_phrase_embedding(sentence1, self.word_vectors, self.embedding_dim)
        emb2 = create_phrase_embedding(sentence2, self.word_vectors, self.embedding_dim)
        
        emb1 = self.pad_sequence(emb1)
        emb2 = self.pad_sequence(emb2)

        return torch.tensor(emb1, dtype=torch.float32), torch.tensor(emb2, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def pad_sequence(self, seq):
        if seq.shape[0] < self.max_len:
            pad_size = self.max_len - seq.shape[0]
            return np.vstack([seq, np.zeros((pad_size, self.embedding_dim))])
        else:
            return seq[:self.max_len]

def load_and_preprocess_paws_dataset(word_vectors, embedding_dim):
    dataset = load_dataset("google-research-datasets/paws", "unlabeled_final", split="train")
    sentence_pairs = list(zip(dataset['sentence1'], dataset['sentence2']))
    labels = dataset['label']
    return sentence_pairs, labels

sentence_pairs, labels = load_and_preprocess_paws_dataset(word_vectors, embedding_dim)
X_train, X_temp, y_train, y_temp = train_test_split(sentence_pairs, labels, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = SentencePairDataset(X_train, y_train, word_vectors, embedding_dim)
dev_dataset = SentencePairDataset(X_dev, y_dev, word_vectors, embedding_dim)
test_dataset = SentencePairDataset(X_test, y_test, word_vectors, embedding_dim)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_and_evaluate_model(model, train_loader, dev_loader, test_loader, epochs=10, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for emb1, emb2, label in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(emb1, emb2).squeeze()
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for emb1, emb2, label in tqdm(test_loader):
            outputs = model(emb1, emb2).squeeze()
            y_true.extend(label.tolist())
            y_pred.extend(outputs.sigmoid().round().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")

class PhraseSimilarityModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, rnn_type='LSTM', bidirectional=True, num_layers=2):
        super(PhraseSimilarityModel, self).__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError("Invalid rnn_type. Choose either 'LSTM' or 'GRU'.")

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, emb1, emb2):
        _, (h_n1, _) = self.rnn(emb1)
        _, (h_n2, _) = self.rnn(emb2)

        if self.bidirectional:
            h_n1 = h_n1.view(self.num_layers, 2, -1, self.hidden_dim)[-1]
            h_n2 = h_n2.view(self.num_layers, 2, -1, self.hidden_dim)[-1]
            h_n1 = torch.cat((h_n1[0], h_n1[1]), dim=1)
            h_n2 = torch.cat((h_n2[0], h_n2[1]), dim=1)
        else:
            h_n1 = h_n1[-1]
            h_n2 = h_n2[-1]

        combined = torch.abs(h_n1 - h_n2)
        output = self.fc(combined)
        return output

hidden_dim = 64
num_layers = 2
model = PhraseSimilarityModel(embedding_dim, hidden_dim, rnn_type='LSTM', bidirectional=True, num_layers=num_layers)
train_and_evaluate_model(model, train_loader, dev_loader, test_loader, epochs=10, learning_rate=0.001)
