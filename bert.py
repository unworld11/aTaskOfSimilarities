import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Load the PAWS dataset
ds = load_dataset("google-research-datasets/paws", "labeled_final")

# Split the dataset into train and validation sets
train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(
    ds['train']['sentence1'], ds['train']['sentence2'], ds['train']['label'],
    test_size=0.1, random_state=42
)

# Initialize tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the input texts as sentence pairs
train_encodings = tokenizer(train_texts1, train_texts2, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts1, val_texts2, truncation=True, padding=True, max_length=128)

# Create PyTorch datasets
class SentenceSimilarityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentenceSimilarityDataset(train_encodings, train_labels)
val_dataset = SentenceSimilarityDataset(val_encodings, val_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_true.extend(labels.cpu().tolist())

    accuracy = accuracy_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds, average='weighted')
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert_paws")
tokenizer.save_pretrained("./fine_tuned_bert_paws")

print("Fine-tuning completed and model saved!")
