import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Get BERT embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Extract keywords using TF-IDF
def extract_keywords(text, top_n=10):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    try:
        stop_words = list(set(stopwords.words('english')))
    except:
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were']

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    word_scores = {word: score for word, score in zip(feature_names, scores) if score > 0}
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_words[:top_n])

# Compute BERT cosine similarity
def calculate_similarity(student_answer, reference_answer):
    embedding1 = get_bert_embedding(student_answer)
    embedding2 = get_bert_embedding(reference_answer)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

# Compute keyword similarity and missing keywords
def calculate_keyword_similarity(student_answer, reference_answers, critical_keywords=None):
    if critical_keywords is None:
        critical_keywords = []

    # Normalize and clean critical keywords
    critical_keywords_clean = [re.sub(r'[^\w\s]', '', kw.lower().strip()) for kw in critical_keywords]

    # Preprocess student answer for phrase matching
    student_text_clean = re.sub(r'[^\w\s]', '', student_answer.lower())

    # Phrase-level critical keyword match
    critical_missing = [kw for kw in critical_keywords_clean if kw not in student_text_clean]
    matched_critical = len(critical_keywords_clean) - len(critical_missing)
    critical_match_percent = (matched_critical / len(critical_keywords_clean)) * 100 if critical_keywords_clean else 0.0

    # TF-IDF keyword matching for general overlap
    student_keywords = set(extract_keywords(student_answer, top_n=15).keys())
    reference_keywords = set()
    for ref in reference_answers:
        reference_keywords.update(extract_keywords(ref, top_n=15).keys())

    if not reference_keywords:
        return 0.0, [], critical_missing, critical_match_percent

    common_keywords = student_keywords.intersection(reference_keywords)
    missing_keywords = reference_keywords - student_keywords

    keyword_similarity = len(common_keywords) / len(reference_keywords)

    # Apply penalty for missing critical phrases
    penalty = 0.2 * len(critical_missing)
    keyword_similarity = max(0.0, keyword_similarity - penalty)

    return keyword_similarity, list(missing_keywords), critical_missing, critical_match_percent



# Feedback generator
def generate_feedback(similarity, grade, missing_keywords, critical_missing):
    base_feedback = {
        'A': "Excellent answer! Your response matches the expected answer very well.",
        'B': "Good answer. Your response captures most of the key concepts.",
        'C': "Acceptable answer. Consider adding more details or key concepts.",
        'D': "Your answer is on the right track but missing important concepts or details.",
        'F': "Your answer needs significant improvement. Please review the material."
    }[grade]

    notes = ""
    if critical_missing:
        notes += f" Critical keywords missing: {', '.join(critical_missing)}."
    if missing_keywords:
        notes += f" Consider including: {', '.join(missing_keywords[:5])}."

    return base_feedback + notes

# Grading logic
def grade_answer(student_answer, reference_answers, thresholds=None, critical_keywords=None):
    if thresholds is None:
        thresholds = {'A': 0.85, 'B': 0.75, 'C': 0.65, 'D': 0.55, 'F': 0.0}

    best_semantic_similarity = max(calculate_similarity(student_answer, ref) for ref in reference_answers)

    keyword_similarity, missing_keywords, critical_missing, critical_match_percent = calculate_keyword_similarity(
        student_answer, reference_answers, critical_keywords
    )

    combined_similarity = (best_semantic_similarity * 0.7) + (keyword_similarity * 0.3)

    if combined_similarity >= thresholds['A']:
        grade = 'A'
    elif combined_similarity >= thresholds['B']:
        grade = 'B'
    elif combined_similarity >= thresholds['C']:
        grade = 'C'
    elif combined_similarity >= thresholds['D']:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'semantic_similarity': best_semantic_similarity,
        'keyword_similarity': keyword_similarity,
        'combined_similarity': combined_similarity,
        'grade': grade,
        'missing_keywords': missing_keywords,
        'critical_missing': critical_missing,
        'critical_match_percent': critical_match_percent,
        'feedback': generate_feedback(combined_similarity, grade, missing_keywords, critical_missing)
    }

# Gradio UI callback
def gradio_answer_grader(question, student_answer, reference_answers, critical_keywords_input):
    ref_answers_list = [ans.strip() for ans in reference_answers.split('||')]
    critical_keywords = [kw.strip().lower() for kw in critical_keywords_input.split(',') if kw.strip()]
    result = grade_answer(student_answer, ref_answers_list, critical_keywords=critical_keywords)

    return (
        f"Semantic Similarity: {result['semantic_similarity']:.4f}\n"
        f"Keyword Similarity: {result['keyword_similarity']:.4f}\n"
        f"Important Keywords Match: {result['critical_match_percent']:.2f}%\n"
        f"Combined Score: {result['combined_similarity']:.4f}\n"
        f"Grade: {result['grade']}\n"
        f"Feedback: {result['feedback']}"
    )

# Launch Gradio interface
iface = gr.Interface(
    fn=gradio_answer_grader,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Student Answer"),
        gr.Textbox(label="Reference Answers (separate with ||)"),
        gr.Textbox(label="High-Priority Keywords (comma-separated)")
    ],
    outputs=gr.Textbox(label="Grading Result"),
    title="BERT Answer Grading System with Critical Keyword Weighting",
    description="Grade student answers using semantic similarity and keyword analysis, with optional high-priority keyword penalties."
)

if __name__ == "__main__":
    iface.launch()
