import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Initialize models for text processing
print("Loading NLP models...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("Loaded BERT tokenizer.")
model = AutoModel.from_pretrained('bert-base-uncased')
print("Loaded BERT model.")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded SBERT model.")

# -------- UTILITY FUNCTIONS --------

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

# Load data
print("Loading data...")
csv_file_path = r"ModelTraining\AutoGradeAI_enhanced_dataset2_inflated.csv"
df = pd.read_csv(csv_file_path)

# Feature extraction using only BERT and SBERT similarities
print("Extracting features using NLP models...")
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
    
    # Store features and label (only BERT and SBERT similarities)
    features.append([bert_similarity, sbert_similarity])
    labels.append(ai_score)
    
    # Print progress
    if index % 10 == 0:
        print(f"Processed {index} rows...")

X = np.array(features)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

# Quick evaluation
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest (No Keywords) - MSE: {mse:.4f}, R²: {r2:.4f}")

# Save the model with a different name to distinguish from the keyword version
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "random_forest_grading_model_no_keywords.joblib")
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

print("Training complete!")
