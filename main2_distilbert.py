import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset
train_df = pd.read_csv('./train-dataset_clean.csv')
train_df = train_df.drop(['id'], axis=1)
train_df['label'] = train_df['label'].map({'CG': 1, 'OR': 0}).astype(int)

# Function to clean the reviews
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only alphabets and spaces
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

# Clean the reviews
train_df['review'] = train_df['review'].apply(clean_text)

# Prepare data for training
X = train_df["review"].tolist()
Y = train_df["label"].tolist()

# Load DistilBERT tokenizer
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_tokenizer_f1_0.8755')

# Tokenize the input data
inputs = tokenizer(X, padding=True, truncation=True, return_tensors='pt', max_length=256, return_attention_mask=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(inputs['input_ids'], Y, test_size=0.2, random_state=42, stratify=Y)

# Create Dataset objects
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewsDataset({'input_ids': X_train}, y_train)
val_dataset = ReviewsDataset({'input_ids': X_val}, y_val)

# Load the pre-trained DistilBERT model
#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model_f1_0.8755', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    seed=42
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        'roc_auc': roc_auc_score(p.label_ids, p.predictions[:, 1])
    }
)

# Train the model
trainer.train()

# Evaluate the model
predictions, labels, _ = trainer.predict(val_dataset)
y_pred = np.argmax(predictions, axis=1)

# Generate evaluation metrics
classification_report_result = classification_report(y_val, y_pred)
confusion_matrix_result = confusion_matrix(y_val, y_pred)
roc_auc = roc_auc_score(y_val, predictions[:, 1])
f1 = f1_score(y_val, y_pred)
f1_str = f"{f1:.4f}"

# Print results
print("Validation Results:")
print(classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)
print(f"ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}\n")

# Save the best model and tokenizer
model.save_pretrained(f'./distilbert_model_f1_{f1_str}')
tokenizer.save_pretrained(f'./distilbert_tokenizer_f1_{f1_str}')
print("Best model and tokenizer saved successfully!")
