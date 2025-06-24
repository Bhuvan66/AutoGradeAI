import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
import yake
from keybert import KeyBERT
import matplotlib.pyplot as plt

# Make sure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load common words from JSON if available, otherwise use default stopwords
COMMON_WORDS = set()
common_words_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common_words.json")
if os.path.exists(common_words_path):
    with open(common_words_path, "r", encoding="utf-8") as f:
        COMMON_WORDS = set(json.load(f))
COMMON_WORDS = COMMON_WORDS.union(set(stopwords.words('english')))

# Initialize models for text processing
print("Loading NLP models...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("Loaded BERT tokenizer.")
model = AutoModel.from_pretrained('bert-base-uncased')
print("Loaded BERT model.")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded SBERT model.")
kw_model = KeyBERT()
print("Loaded KeyBERT model.")
yake_extractor = yake.KeywordExtractor()
print("Loaded YAKE extractor.")

# Constants for keyword extraction
KEYBERT_TOP_N = 8

# -------- UTILITY FUNCTIONS --------

def clean_keyword(keyword):
    """Clean a keyword by removing non-alphanumeric characters and checking validity."""
    keyword = re.sub(r'[^\w\s]', '', keyword.lower().strip())
    if len(keyword) < 3 or keyword in COMMON_WORDS:
        return None
    if keyword.isdigit():
        return None
    return keyword

def extract_keywords_priority(text):
    """Extract keywords from text and categorize them by priority."""
    keybert_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=12,
        top_n=KEYBERT_TOP_N
    )
    cleaned_keywords = []
    seen = set()
    for keyword, score in keybert_keywords:
        cleaned = clean_keyword(keyword)
        if cleaned and cleaned not in seen:
            cleaned_keywords.append((cleaned, score))
            seen.add(cleaned)
    sorted_keywords = sorted(cleaned_keywords, key=lambda x: -x[1])
    total = len(sorted_keywords)
    if total == 0:
        return [], [], []
    high_n = max(1, int(total * 0.3))
    med_n = max(1, int(total * 0.4))
    low_n = total - high_n - med_n
    high = [k for k, _ in sorted_keywords[:high_n]]
    medium = [k for k, _ in sorted_keywords[high_n:high_n+med_n]]
    low = [k for k, _ in sorted_keywords[high_n+med_n:]]
    return high, medium, low

def get_bert_embedding(text):
    """Get BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def calculate_similarity(student_answer, reference_answer):
    """Calculate BERT-based similarity between two texts."""
    embedding1 = get_bert_embedding(student_answer)
    embedding2 = get_bert_embedding(reference_answer)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

def calculate_sbert_similarity(student_answer, reference_answer):
    """Calculate SBERT-based similarity between two texts."""
    emb1 = sbert_model.encode([student_answer])[0]
    emb2 = sbert_model.encode([reference_answer])[0]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return sim

def calculate_keyword_similarity(student_answer, reference_answers, priority_keywords=None):
    """Calculate keyword-based similarity with priority weighting."""
    if priority_keywords is None:
        priority_keywords = {'high': [], 'medium': [], 'low': []}

    student_text_clean = re.sub(r'[^\w\s]', '', student_answer.lower())

    matched = 0
    critical_missing = []
    missing_keywords = []

    weights = {'high': 3, 'medium': 2, 'low': 1}
    total_possible_weight = 0

    for priority, keywords in priority_keywords.items():
        for kw in keywords:
            total_possible_weight += weights[priority]
            if kw in student_text_clean:
                matched += weights[priority]
            else:
                critical_missing.append(kw)

    keyword_similarity = matched / total_possible_weight if total_possible_weight else 0.0

    return keyword_similarity, missing_keywords, critical_missing, keyword_similarity * 100

# -------- NEURAL NETWORK MODEL --------

class GradingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GradingNN(nn.Module):
    def __init__(self, input_size=3):
        super(GradingNN, self).__init__()
        # Enhanced network with more layers and neurons
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

# -------- MAIN TRAINING SCRIPT --------

# Path to the CSV file
csv_file_path = r"ModelTraining\AutoGradeAI_enhanced_dataset4.csv"

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv(csv_file_path)

# Extract features for training
print("Extracting features for training...")
features = []
labels = []

for index, row in df.iterrows():
    student_answer = row['Student Answer']
    preferred_answer = row['Preferred Answer']
    ai_score = row['AI Score']
    
    # Calculate BERT similarity
    bert_similarity = calculate_similarity(student_answer, preferred_answer)
    
    # Calculate SBERT similarity
    sbert_similarity = calculate_sbert_similarity(student_answer, preferred_answer)
    
    # Extract keywords and calculate keyword similarity
    high, medium, low = extract_keywords_priority(preferred_answer)
    priority_keywords = {'high': high, 'medium': medium, 'low': low}
    keyword_similarity, _, _, _ = calculate_keyword_similarity(
        student_answer, [preferred_answer], priority_keywords
    )
    
    # Store features and label
    features.append([bert_similarity, sbert_similarity, keyword_similarity])
    labels.append(ai_score)
    
    # Print progress
    if index % 10 == 0:
        print(f"Processed {index} rows...")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets and dataloaders
train_dataset = GradingDataset(X_train, y_train)
test_dataset = GradingDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GradingNN().to(device)
# Using a weighted MSE loss to penalize larger errors more heavily
criterion = nn.MSELoss()
# Reduced learning rate for better convergence and weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
# More aggressive scheduler with early stopping
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

# Training loop
num_epochs = 200  # Increased epochs
print(f"\nTraining neural network for {num_epochs} epochs...")

train_losses = []
val_losses = []
val_r2_scores = []

# Add early stopping
best_val_loss = float('inf')
patience = 15
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    
    # Calculate R2 score
    r2 = r2_score(all_labels, all_preds)
    val_r2_scores.append(r2)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {r2:.4f} (Best)")
    else:
        patience_counter += 1
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {r2:.4f}")
    
    # If patience exceeded, stop training
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model from training")

print("Training completed!")

# Final evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
mse = mean_squared_error(all_labels, all_preds)
r2 = r2_score(all_labels, all_preds)
abs_errors = np.abs(np.array(all_labels) - np.array(all_preds))
mean_abs_error = np.mean(abs_errors)
median_abs_error = np.median(abs_errors)

print("\nNeural Network Final Test Results:")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Mean Absolute Error: {mean_abs_error:.4f}")
print(f"Median Absolute Error: {median_abs_error:.4f}")
print(f"Max Absolute Error: {np.max(abs_errors):.4f}")
print(f"% of predictions within 0.05 of true value: {(abs_errors < 0.05).mean() * 100:.1f}%")
print(f"% of predictions within 0.1 of true value: {(abs_errors < 0.1).mean() * 100:.1f}%")

# Save the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(parent_dir, "trained_nn_grading_model.pt")
torch.save(model.state_dict(), model_path)
print(f"Neural network model saved to {model_path}")

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_r2_scores, label='R² Score')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.title('Validation R² Score')
plt.legend()

# Save the plot
plot_path = os.path.join(parent_dir, "nn_training_progress.png")
plt.tight_layout()
plt.savefig(plot_path)
print(f"Training progress plot saved to {plot_path}")

# Plot prediction vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(all_labels, all_preds, alpha=0.6)

# Add perfect prediction line
min_val = min(min(all_labels), min(all_preds))
max_val = max(max(all_labels), max(all_preds))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Neural Network: Predicted vs Actual Scores (Test Set)')
plt.grid(True, alpha=0.3)

# Save the plot
nn_plot_path = os.path.join(parent_dir, "nn_prediction_vs_actual.png")
plt.savefig(nn_plot_path)
print(f"Saved prediction vs actual plot to {nn_plot_path}")

# Create a class to easily use the trained model
class NeuralNetworkGrader:
    def __init__(self, model_path):
        self.model = GradingNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def predict_score(self, bert_similarity, sbert_similarity, keyword_similarity):
        """Predict score using the neural network model"""
        features = torch.FloatTensor([[bert_similarity, sbert_similarity, keyword_similarity]])
        with torch.no_grad():
            score = self.model(features).item()
        return max(0.0, min(1.0, score))

# Test the neural network on sample inputs
print("\nNeural Network predictions for sample inputs:")
sample_inputs = [
    [0.9, 0.9, 0.9],  # Excellent match
    [0.8, 0.7, 0.6],  # Good match
    [0.6, 0.5, 0.5],  # Average match
    [0.4, 0.3, 0.2],  # Below average match
    [0.2, 0.1, 0.1]   # Poor match
]

model.eval()
for inputs in sample_inputs:
    with torch.no_grad():
        prediction = model(torch.FloatTensor([inputs]).to(device)).item()
    print(f"BERT: {inputs[0]:.2f}, SBERT: {inputs[1]:.2f}, KW: {inputs[2]:.2f} => Score: {prediction:.4f}")

# Save a sample usage script
nn_usage_script = """
from neural_network_grader import NeuralNetworkGrader

# Initialize the grader
grader = NeuralNetworkGrader("trained_nn_grading_model.pt")

# Use the grader to predict scores
score = grader.predict_score(
    bert_similarity=0.85,   # Semantic similarity from BERT
    sbert_similarity=0.78,  # Sentence similarity from SBERT
    keyword_similarity=0.92 # Keyword matching score
)

print(f"Predicted score: {score:.2f}")
"""

nn_usage_path = os.path.join(parent_dir, "neural_network_grader.py")
with open(nn_usage_path, "w") as f:
    f.write(nn_usage_script)

print(f"Sample usage script saved to {nn_usage_path}")
print("\nDone!")
