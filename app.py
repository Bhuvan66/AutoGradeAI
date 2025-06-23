import gradio as gr
import torch
import numpy as np
import re
import os
import joblib
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import json

# Make sure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load common words from JSON if available, otherwise use default stopwords
COMMON_WORDS = set()
common_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_words.json")
if os.path.exists(common_words_path):
    with open(common_words_path, "r", encoding="utf-8") as f:
        COMMON_WORDS = set(json.load(f))
COMMON_WORDS = COMMON_WORDS.union(set(stopwords.words('english')))

# Load models
print("Loading NLP models...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()

# Load the trained grading model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_grading_model.joblib")
grading_model = joblib.load(model_path)

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

def predict_score(bert_similarity, sbert_similarity, keyword_similarity):
    """
    Predict a score using the polynomial model which generally performs better than linear models.
    """
    # Prepare input features for prediction
    features = np.array([[bert_similarity, sbert_similarity, keyword_similarity]])
    
    # Polynomial regression transforms inputs before prediction
    predicted_score = grading_model.predict(features)[0]
    
    # Ensure the score is between 0 and 1
    predicted_score = max(0.0, min(1.0, predicted_score))
    
    return predicted_score

def grade_answer(student_answer, reference_answer):
    """Grade the student answer based on the reference answer."""
    if not student_answer.strip() or not reference_answer.strip():
        return {
            "score": 0.0,
            "bert_similarity": 0.0,
            "sbert_similarity": 0.0,
            "keyword_similarity": 0.0,
            "missing_keywords": [],
            "feedback": "Answer or reference cannot be empty."
        }
    
    # Calculate BERT similarity
    bert_similarity = calculate_similarity(student_answer, reference_answer)
    
    # Calculate SBERT similarity
    sbert_similarity = calculate_sbert_similarity(student_answer, reference_answer)
    
    # Extract keywords and calculate keyword similarity
    high, medium, low = extract_keywords_priority(reference_answer)
    priority_keywords = {'high': high, 'medium': medium, 'low': low}
    keyword_similarity, _, missing_kw, _ = calculate_keyword_similarity(
        student_answer, [reference_answer], priority_keywords
    )
    
    # Predict score
    score = predict_score(bert_similarity, sbert_similarity, keyword_similarity)
    
    # Generate feedback based on missing keywords
    feedback = ""
    if score < 0.4:
        feedback = "Your answer is missing key concepts. Consider addressing: " + ", ".join(high)
    elif score < 0.7:
        feedback = "Good attempt, but try to include more of these concepts: " + ", ".join(missing_kw[:3])
    else:
        feedback = "Excellent answer! You've covered the main concepts well."
    
    # Format score as percentage
    score_percent = score * 100
    
    return {
        "score": f"{score_percent:.1f}%",
        "bert_similarity": f"{bert_similarity:.2f}",
        "sbert_similarity": f"{sbert_similarity:.2f}",
        "keyword_similarity": f"{keyword_similarity:.2f}",
        "missing_keywords": ", ".join(missing_kw[:5]) if missing_kw else "None",
        "feedback": feedback
    }

# Create Gradio interface
def ui_grade_answer(student_answer, reference_answer):
    results = grade_answer(student_answer, reference_answer)
    
    # Create formatted output
    output = f"""
## Grading Results

**Score:** {results['score']}

### Similarity Metrics
- BERT Similarity: {results['bert_similarity']}
- SBERT Similarity: {results['sbert_similarity']}
- Keyword Similarity: {results['keyword_similarity']}

### Feedback
{results['feedback']}

### Missing Keywords
{results['missing_keywords']}
    """
    
    return output

# Create and launch the interface
iface = gr.Interface(
    fn=ui_grade_answer,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter the student's answer here...", label="Student Answer"),
        gr.Textbox(lines=5, placeholder="Enter the reference/preferred answer here...", label="Reference Answer")
    ],
    outputs=gr.Markdown(),
    title="AutoGradeAI - Automatic Answer Grading",
    description="Enter a student answer and a reference answer to get an automatic grade and feedback.",
    theme="huggingface",
    examples=[
        ["The water cycle is the process where water circulates between the earth's oceans, atmosphere, and land. It involves evaporation, condensation, and precipitation.", 
         "The water cycle, also known as the hydrologic cycle, is the continuous movement of water on, above, and below the surface of the Earth. Water changes between liquid, vapor, and ice states through processes of evaporation, transpiration, condensation, precipitation, and runoff."],
        ["Photosynthesis is when plants make food using sunlight.", 
         "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. In plants, photosynthesis occurs in chloroplasts and generates oxygen as a byproduct."]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)
    print("Gradio app is running!")
