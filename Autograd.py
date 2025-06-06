import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import math

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

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

def calculate_similarity(student_answer, reference_answer):
    embedding1 = get_bert_embedding(student_answer)
    embedding2 = get_bert_embedding(reference_answer)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

def parse_priority_keywords(raw_input):
    priority_keywords = {'high': [], 'medium': [], 'low': []}
    lines = raw_input.lower().split('\n')
    for line in lines:
        if ':' in line:
            level, words = line.split(':', 1)
            level = level.strip()
            keywords = [re.sub(r'[^\w\s]', '', kw.strip()) for kw in words.split(',')]
            if level in priority_keywords:
                priority_keywords[level].extend(keywords)
    return priority_keywords

def calculate_keyword_similarity(student_answer, reference_answers, priority_keywords=None):
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

def generate_feedback(similarity, grade, missing_keywords, critical_missing):
    base_feedback = {
        'A': "Excellent answer! Your response matches the expected answer very well.",
        'B': "Good answer. Your response captures most of the key concepts.",
        'C': "Acceptable answer. Consider adding more details or key concepts.",
        'D': "This answer is incorrect and not on the right track, as it misses key concepts and doesn't align with the core idea of the topic.",
        'F': "Your answer needs significant improvement. Please review the material."
    }[grade]

    notes = ""
    if critical_missing:
        notes += f" Missing key phrases: {', '.join(critical_missing[:5])}."
    return base_feedback + notes

def normalize_score_z(raw_score, mean=0.6, std=0.15):
    """
    Normalize using Z-score, assuming raw_score is between 0 and 1.
    Then scale using a sigmoid-like logistic curve to map to 0–100.
    """
    z = (raw_score - mean) / std
    logistic_scaled = 100 / (1 + math.exp(-z))  # S-shaped curve
    return logistic_scaled

def grade_answer(student_answer, reference_answers, thresholds=None, priority_keywords=None):
    if thresholds is None:
        # Thresholds based on scaled score out of 100
        thresholds = {'A': 85, 'B': 70, 'C': 55, 'D': 40, 'F': 0}

    best_semantic_similarity = max(calculate_similarity(student_answer, ref) for ref in reference_answers)

    keyword_similarity, missing_keywords, critical_missing, critical_match_percent = calculate_keyword_similarity(
        student_answer, reference_answers, priority_keywords
    )

    combined_similarity = (best_semantic_similarity * 0.7) + (keyword_similarity * 0.3)

    # Use new z-normalization approach for bell curve scaling
    bell_score = normalize_score_z(combined_similarity)

    # Assign grade based on scaled bell_score
    if bell_score >= thresholds['A']:
        grade = 'A'
    elif bell_score >= thresholds['B']:
        grade = 'B'
    elif bell_score >= thresholds['C']:
        grade = 'C'
    elif bell_score >= thresholds['D']:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'semantic_similarity': best_semantic_similarity,
        'keyword_similarity': keyword_similarity,
        'combined_similarity': combined_similarity,
        'bell_scaled_score': bell_score,
        'grade': grade,
        'missing_keywords': missing_keywords,
        'critical_missing': critical_missing,
        'critical_match_percent': critical_match_percent,
        'feedback': generate_feedback(combined_similarity, grade, missing_keywords, critical_missing)
    }

def gradio_answer_grader(question, student_answer, reference_answers, priority_keyword_input):
    ref_answers_list = [ans.strip() for ans in reference_answers.split('||')]
    priority_keywords = parse_priority_keywords(priority_keyword_input)
    result = grade_answer(student_answer, ref_answers_list, priority_keywords=priority_keywords)

    return (
        f"Semantic Similarity: {result['semantic_similarity']:.4f}\n"
        f"Keyword Match Score (weighted): {result['keyword_similarity']:.4f}\n"
        f"Weighted Keyword Match Percent: {result['critical_match_percent']:.2f}%\n"
        f"Combined Score (raw): {result['combined_similarity']:.4f}\n"
        f"Combined Score (custom scaled 0-100): {result['bell_scaled_score']:.2f}\n"
        f"Grade: {result['grade']}\n"
        f"Feedback: {result['feedback']}"
    )

iface = gr.Interface(
    fn=gradio_answer_grader,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Student Answer"),
        gr.Textbox(label="Reference Answers (separate with ||)"),
        gr.Textbox(label="Priority Keywords (e.g., high: keyword1, keyword2...)")
    ],
    outputs=gr.Textbox(label="Grading Result"),
    title="Answer Grading System with Keyword Priorities",
    description="Grades student answers using BERT-based semantic similarity and priority-weighted keyword matching."
)

if __name__ == "__main__":
    iface.launch()
