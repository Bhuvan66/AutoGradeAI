import pandas as pd
import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import os
import json
import yake
from keybert import KeyBERT
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_squared_error, r2_score, 
                           mean_absolute_error, median_absolute_error)
import joblib

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

# Load data
print("Loading data...")
csv_file_path = r"ModelTraining\AutoGradeAI_enhanced_dataset2.csv"
df = pd.read_csv(csv_file_path)

# Feature extraction using actual NLP models
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

X = np.array(features)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Polynomial (Deg 4)": Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('ridge', Ridge(alpha=0.5))
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    "MLP":MLPRegressor(
    hidden_layer_sizes=(1500, 1000, 500),  # Added layer
    activation='relu',                   # Changed from default tanh
    solver='adam',                       # Better for medium datasets
    alpha=0.001,                         # Stronger regularization
    learning_rate='adaptive',            # Better convergence
    max_iter=1000,
    early_stopping=True,                 # Prevent overfitting
    validation_fraction=0.1,
    random_state=42
)
}

# Results storage
results = []

# Train and evaluate models
print("\nTraining models...")
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    abs_errors = np.abs(y_test - y_pred)
    metrics = {
        'Model': name,
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred),
        'MaxError': np.max(abs_errors),
        'Within_0.05': (abs_errors < 0.05).mean(),
        'Within_0.1': (abs_errors < 0.1).mean(),
        'Train_Time': time.time() - start_time
    }
    results.append(metrics)
    
    print(f"Completed {name} in {metrics['Train_Time']:.2f}s")

# Create and save results DataFrame
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "model_metrics.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved metrics to {results_csv_path}")

# Identify and save best model
best_model_info = results_df.loc[results_df['R2'].idxmax()]
best_model = models[best_model_info['Model']]
best_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "best_model.joblib")
joblib.dump(best_model, best_model_path)
print(f"Saved best model ({best_model_info['Model']}) to {best_model_path}")

# Display results
print("\nModel Evaluation Results:")
print(results_df.round(4).to_string(index=False))

# Feature importance for tree-based models
print("\nFeature Importances:")
if hasattr(best_model, 'feature_importances_'):
    for name, importance in zip(["BERT", "SBERT", "Keyword"], best_model.feature_importances_):
        print(f"{name}: {importance:.4f}")
elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['ridge'], 'coef_'):
    print("Polynomial feature coefficients available")
else:
    print("Feature importances not available for this model type")

print("\nTraining complete!")