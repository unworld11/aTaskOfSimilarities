import os
import time
from groq import Groq
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from typing import List

# Initialize the Groq client
client = Groq(api_key="your key")

def check_similarity_zero_shot(sentence1: str, sentence2: str) -> int:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a phrase/sentence similarity model. Assess whether the two provided sentences are similar in meaning and structure. Consider synonyms, sentence structure, and overall meaning. Output '1' if they are similar, '0' if they are not."
                },
                {
                    "role": "user",
                    "content": f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences similar? Respond with only 0 or 1."
                }
            ],
            temperature=0,
            max_tokens=1,
            top_p=1,
            stream=False,
            stop=None,
        )
        return int(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in zero-shot similarity check: {e}")
        return -1  # Return a default value or handle it as needed

def check_similarity_few_shot(sentence1: str, sentence2: str) -> int:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a phrase/sentence similarity model. Assess whether the two provided sentences are similar in meaning and structure. Consider synonyms, sentence structure, and overall meaning. Output '1' if they are similar, '0' if they are not."
                },
                {
                    "role": "user",
                    "content": (
                        "Sentence 1: The cat sat on the mat.\n"
                        "Sentence 2: The feline rested on the rug.\n"
                        "Are these sentences similar? Respond with only 0 or 1.\n"
                        "1\n"
                        "Sentence 1: The sky is blue.\n"
                        "Sentence 2: The ocean is deep.\n"
                        "Are these sentences similar? Respond with only 0 or 1.\n"
                        "0\n"
                        f"Sentence 1: {sentence1}\n"
                        f"Sentence 2: {sentence2}\n"
                        "Are these sentences similar? Respond with only 0 or 1."
                    )
                }
            ],
            temperature=0,
            max_tokens=1,
            top_p=1,
            stream=False,
            stop=None,
        )
        return int(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in few-shot similarity check: {e}")
        return -1  # Return a default value or handle it as needed

def validate_on_paws(num_samples: int = 50, few_shot: bool = False) -> None:
    # Load the PAWS dataset
    ds = load_dataset("google-research-datasets/paws", "labeled_final")
    
    # Use a subset of the test set for quicker validation
    test_data = ds['train'].shuffle(seed=42).select(range(num_samples))
    
    predictions: List[int] = []
    true_labels: List[int] = []
    
    for item in tqdm(test_data, desc="Processing samples"):
        try:
            if few_shot:
                pred = check_similarity_few_shot(item['sentence1'], item['sentence2'])
            else:
                pred = check_similarity_zero_shot(item['sentence1'], item['sentence2'])
            if pred != -1:  # Only append valid predictions
                predictions.append(pred)
                true_labels.append(item['label'])
            time.sleep(1)  # Add a delay to avoid hitting API rate limits
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Debugging: Print mismatched predictions
    for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
        if pred != true_label:
            print(f"Mismatch at index {i}: Prediction: {pred}, True Label: {true_label}, Sentence1: {test_data[i]['sentence1']}, Sentence2: {test_data[i]['sentence2']}")

# Run the validation
print("Validating the model on PAWS dataset with zero-shot setting...")
validate_on_paws(num_samples=50, few_shot=False)

print("Validating the model on PAWS dataset with few-shot setting...")
validate_on_paws(num_samples=50, few_shot=True)
