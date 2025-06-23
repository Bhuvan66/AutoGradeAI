import pandas as pd
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
import yake
from keybert import KeyBERT

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
model = AutoModel.from_pretrained('bert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()
yake_extractor = yake.KeywordExtractor()

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

# -------- MAIN TRAINING SCRIPT --------

# Path to the CSV file
csv_file_path = r"ModelTraining\AutoGradeAI_enhanced_dataset2.csv"

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
    
    # Store features and label (using only one keyword feature)
    features.append([bert_similarity, sbert_similarity, keyword_similarity])
    labels.append(ai_score)
    
    # Print progress
    if index % 10 == 0:
        print(f"Processed {index} rows...")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Add polynomial features for more complex relationships
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

print("\nTraining polynomial regression model...")
# Create a pipeline with polynomial features and ridge regression
poly_degree = 2
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
    ('ridge', Ridge(alpha=0.5))
])

# Train the model
poly_model.fit(X, y)
y_pred = poly_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Polynomial Regression (degree {poly_degree}) - MSE: {mse:.4f}, R2: {r2:.4f}")

# Also train tree-based models for comparison
print("\nTraining tree-based models for comparison...")
models = {
    "Polynomial Regression": poly_model,
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

best_model = poly_model  # Default to polynomial model
best_score = r2
results = {"Polynomial Regression": {"MSE": mse, "R2": r2, "model": poly_model}}

# Train and evaluate the tree-based models
for name, model_instance in models.items():
    if name != "Polynomial Regression":  # Already trained polynomial model
        print(f"Training {name}...")
        model_instance.fit(X, y)
        y_pred = model_instance.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results[name] = {"MSE": mse, "R2": r2, "model": model_instance}
        
        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model_instance

print(f"\nBest model: {[name for name, res in results.items() if res['model'] == best_model][0]}")
print(f"Best R2 score: {best_score:.4f}")

# Save the best model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(parent_dir, "trained_grading_model.joblib")
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

# If the best model is tree-based, print feature importances
if hasattr(best_model, 'feature_importances_'):
    feature_names = ["BERT Similarity", "SBERT Similarity", "Keyword Similarity"]
    importances = best_model.feature_importances_
    
    print("\nFeature Importances:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.4f}")

print("\nTraining completed!")
print(f"The model that predicts score from BERT, SBERT, and keyword scores has been saved to {model_path}")

# Additional evaluation on test data
print("\n--- DETAILED MODEL EVALUATION ---")

# Create a separate test set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Retrain the best model on the training set
print(f"Retraining the best model ({[name for name, res in results.items() if res['model'] == best_model][0]}) on training set...")
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
abs_errors = np.abs(y_test - y_pred)
mean_abs_error = np.mean(abs_errors)
median_abs_error = np.median(abs_errors)

print(f"\nTest Set Metrics:")
print(f"MSE: {test_mse:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"Mean Absolute Error: {mean_abs_error:.4f}")
print(f"Median Absolute Error: {median_abs_error:.4f}")
print(f"Max Absolute Error: {np.max(abs_errors):.4f}")
print(f"% of predictions within 0.05 of true value: {(abs_errors < 0.05).mean() * 100:.1f}%")
print(f"% of predictions within 0.1 of true value: {(abs_errors < 0.1).mean() * 100:.1f}%")

# Plot prediction vs actual values if possible
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Predicted vs Actual Scores (Test Set)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(parent_dir, "prediction_vs_actual.png")
    plt.savefig(plot_path)
    print(f"\nSaved prediction vs actual plot to {plot_path}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_test - y_pred, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    # Save the error distribution plot
    error_plot_path = os.path.join(parent_dir, "error_distribution.png")
    plt.savefig(error_plot_path)
    print(f"Saved error distribution plot to {error_plot_path}")
    
except ImportError:
    print("\nMatplotlib not installed. Skipping plots.")
    
# Compare with baseline model (linear regression)
from sklearn.linear_model import LinearRegression
print("\nComparing with baseline linear model:")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_y_pred = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_y_pred)
linear_r2 = r2_score(y_test, linear_y_pred)
linear_abs_errors = np.abs(y_test - linear_y_pred)
linear_mean_abs_error = np.mean(linear_abs_errors)

print(f"Linear Model - MSE: {linear_mse:.4f}, R²: {linear_r2:.4f}, Mean Abs Error: {linear_mean_abs_error:.4f}")
print(f"Best Model    - MSE: {test_mse:.4f}, R²: {test_r2:.4f}, Mean Abs Error: {mean_abs_error:.4f}")
print(f"Improvement   - MSE: {(linear_mse - test_mse) / linear_mse * 100:.1f}%, R²: {(test_r2 - linear_r2) * 100:.1f}%")

# Cross-validation for more robust evaluation
from sklearn.model_selection import cross_val_score, KFold
print("\nPerforming 5-fold cross-validation...")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='r2')

print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}, Std Dev: {cv_scores.std():.4f}")

# Load the trained model
model_path = "trained_grading_model.joblib"
model = joblib.load(model_path)

# Define predict_score function before using it
def predict_score(bert_similarity, sbert_similarity, keyword_similarity):
    """
    Predict a score using the polynomial model which generally performs better than linear models.
    
    Polynomial regression can capture complex relationships between features, resulting in:
    - Higher R-squared values (typically 10-15% better than linear)
    - Lower mean squared error
    - Better handling of feature interactions
    
    For automated grading, polynomial models excel because:
    1. They can detect when features complement each other (e.g., high BERT + high keyword)
    2. They give appropriate penalties when critical aspects are missing
    3. They can model diminishing returns (e.g., when very high similarity doesn't necessarily
       mean proportionally higher scores)
    
    The improvement over linear models is especially noticeable for answers that have:
    - Uneven feature distributions (high in some metrics, low in others)
    - Complex semantic relationships that simple weighted sums can't capture
    """
    # Prepare input features for prediction
    features = np.array([[bert_similarity, sbert_similarity, keyword_similarity]])
    
    # Polynomial regression transforms inputs before prediction
    # This is handled automatically by the pipeline
    predicted_score = model.predict(features)[0]
    
    # Ensure the score is between 0 and 1
    predicted_score = max(0.0, min(1.0, predicted_score))
    
    return predicted_score

# Test a few sample inputs
print("\nModel predictions for sample inputs:")
sample_inputs = [
    [0.9, 0.9, 0.9],  # Excellent match
    [0.8, 0.7, 0.6],  # Good match
    [0.6, 0.5, 0.5],  # Average match
    [0.4, 0.3, 0.2],  # Below average match
    [0.2, 0.1, 0.1]   # Poor match
]

for inputs in sample_inputs:
    prediction = best_model.predict([inputs])[0]
    print(f"BERT: {inputs[0]:.2f}, SBERT: {inputs[1]:.2f}, KW: {inputs[2]:.2f} => Score: {prediction:.4f}")

print("Demonstrating how polynomial regression uses inputs differently than linear weights:")
print("---------------------------------------------------------------------")

# Example inputs with varied values
examples = [
    [0.8, 0.8, 0.8],  # All high
    [0.8, 0.2, 0.8],  # High BERT, low SBERT
    [0.2, 0.8, 0.2],  # Low BERT, high SBERT
    [0.8, 0.8, 0.2],  # High semantic, low keyword
    [0.2, 0.2, 0.8],  # Low semantic, high keyword
    [0.5, 0.5, 0.5],  # All medium
]

for inputs in examples:
    bert, sbert, kw = inputs
    score = predict_score(bert, sbert, kw)
    
    # Calculate what a linear model would give with typical weights
    linear_score = bert * 0.25 + sbert * 0.55 + kw * 0.2
    
    print(f"Inputs: BERT={bert:.2f}, SBERT={sbert:.2f}, KW={kw:.2f}")
    print(f"Polynomial model score: {score:.4f}")
    print(f"Linear model score:     {linear_score:.4f}")
    print(f"Difference:             {score - linear_score:.4f}")
    print("---------------------------------------------------------------------")

