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
import pandas as pd
from sentence_transformers import SentenceTransformer  # Added import
import os
import json

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Added SBERT model

# --- Keyword Extraction Logic (moved from image model) ---

import yake
from keybert import KeyBERT

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load common words from JSON and combine with stopwords
COMMON_WORDS = set()
common_words_path = os.path.join(os.path.dirname(__file__), "common_words.json")
if os.path.exists(common_words_path):
    with open(common_words_path, "r", encoding="utf-8") as f:
        COMMON_WORDS = set(json.load(f))
COMMON_WORDS = COMMON_WORDS.union(set(stopwords.words('english')))

kw_model = KeyBERT()
yake_extractor = yake.KeywordExtractor()

KEYBERT_TOP_N = 8

def clean_keyword(keyword):
    keyword = re.sub(r'[^\w\s]', '', keyword.lower().strip())
    if len(keyword) < 3 or keyword in COMMON_WORDS:
        return None
    if keyword.isdigit():
        return None
    return keyword

def extract_keywords_priority(text):
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

last_auto_keywords = {"high": [], "medium": [], "low": []}

def generate_auto_keywords(reference_answer):
    global last_auto_keywords
    if not reference_answer:
        last_auto_keywords = {"high": [], "medium": [], "low": []}
        return "Please provide text to generate keywords."
    ref_high, ref_med, ref_low = extract_keywords_priority(reference_answer)
    last_auto_keywords = {
        "high": ref_high,
        "medium": ref_med,
        "low": ref_low
    }
    result = (
        f"High Priority: {', '.join(ref_high)}\n"
        f"Medium Priority: {', '.join(ref_med)}\n"
        f"Low Priority: {', '.join(ref_low)}"
    )
    return result

def process_manual_keywords_v2(high_text, medium_text, low_text):
    global last_auto_keywords
    def parse_section(text):
        return [clean_keyword(kw) for kw in text.split(",") if clean_keyword(kw)]
    manual_high = parse_section(high_text)
    manual_medium = parse_section(medium_text)
    manual_low = parse_section(low_text)
    merged = {
        "high": list(last_auto_keywords["high"]),
        "medium": list(last_auto_keywords["medium"]),
        "low": list(last_auto_keywords["low"])
    }
    for kw in manual_high:
        if kw and kw not in merged["high"]:
            merged["high"].append(kw)
    for kw in manual_medium:
        if kw and kw not in merged["medium"]:
            merged["medium"].append(kw)
    for kw in manual_low:
        if kw and kw not in merged["low"]:
            merged["low"].append(kw)
    result = (
        f"High Priority: {', '.join(merged['high'])}\n"
        f"Medium Priority: {', '.join(merged['medium'])}\n"
        f"Low Priority: {', '.join(merged['low'])}"
    )
    last_auto_keywords.update(merged)
    return result

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

def calculate_sbert_similarity(student_answer, reference_answer):
    emb1 = sbert_model.encode([student_answer])[0]
    emb2 = sbert_model.encode([reference_answer])[0]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return sim

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
           thresholds = {'A': 90, 'B': 80, 'C': 65, 'D': 25, 'F': 0}

    best_semantic_similarity = max(calculate_similarity(student_answer, ref) for ref in reference_answers)

    # Calculate SBERT similarity (not used for grading, just for output)
    best_sbert_similarity = max(calculate_sbert_similarity(student_answer, ref) for ref in reference_answers)

    keyword_similarity, missing_keywords, critical_missing, critical_match_percent = calculate_keyword_similarity(
        student_answer, reference_answers, priority_keywords
    )

   
    combined_similarity = (best_semantic_similarity * 0.25)+(best_sbert_similarity * 0.55) + (keyword_similarity * 0.2)

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
        'sbert_similarity': best_sbert_similarity,  # Added SBERT similarity to result
        'keyword_similarity': keyword_similarity,
        'combined_similarity': combined_similarity,
        'bell_scaled_score': bell_score,
        'grade': grade,
        'missing_keywords': missing_keywords,
        'critical_missing': critical_missing,
        'critical_match_percent': critical_match_percent,
        'feedback': generate_feedback(combined_similarity, grade, missing_keywords, critical_missing)
    }

def gradio_answer_grader(student_answer, reference_answers, priority_keyword_input):
    ref_answers_list = [ans.strip() for ans in reference_answers.split('||')]
    priority_keywords = parse_priority_keywords(priority_keyword_input)
    result = grade_answer(student_answer, ref_answers_list, priority_keywords=priority_keywords)

    return (
        f"Semantic Similarity (BERT): {result['semantic_similarity']:.4f}\n"
        f"Semantic Similarity (SBERT all-MiniLM-L6-v2): {result['sbert_similarity']:.4f}\n"
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

# --- Keyword Extraction UI (added after grading interface) ---
with gr.Blocks(title="Keyword Extraction Tool") as kw_demo:
    gr.Markdown("## Keyword Extraction (from Reference Text)")
    reference_text = gr.Textbox(label="Reference Text", lines=5)
    auto_keywords_btn = gr.Button("Generate Keywords Automatically", variant="secondary")
    keywords_output = gr.Textbox(label="Keywords Output", interactive=False, lines=6)
    manual_keywords_btn = gr.Button("Add Keywords Manually", variant="secondary")
    with gr.Row(visible=False) as manual_input_row:
        manual_high_input = gr.Textbox(
            label="High Priority Keywords (comma separated)",
            placeholder="e.g., process, input, output"
        )
        manual_medium_input = gr.Textbox(
            label="Medium Priority Keywords (comma separated)",
            placeholder="e.g., data, flow"
        )
        manual_low_input = gr.Textbox(
            label="Low Priority Keywords (comma separated)",
            placeholder="e.g., system, label"
        )
    submit_manual_btn = gr.Button("Submit Manual Keywords", visible=False, variant="secondary")

    def show_manual_input():
        return (
            gr.update(visible=True),  # manual_input_row
            gr.update(visible=True)   # submit_manual_btn
        )

    def hide_manual_input():
        return (
            gr.update(visible=False),  # manual_input_row
            gr.update(visible=False)   # submit_manual_btn
        )

    auto_keywords_btn.click(
        generate_auto_keywords,
        inputs=[reference_text],
        outputs=keywords_output
    ).then(
        hide_manual_input,
        outputs=[manual_input_row, submit_manual_btn]
    )

    manual_keywords_btn.click(
        show_manual_input,
        outputs=[manual_input_row, submit_manual_btn]
    )

    submit_manual_btn.click(
        process_manual_keywords_v2,
        inputs=[manual_high_input, manual_medium_input, manual_low_input],
        outputs=keywords_output
    )

if __name__ == "__main__":
    with gr.Blocks(title="Answer Grading and Keyword Extraction Tool") as app:
        gr.Markdown("# Answer Grading System with Keyword Priorities")
        gr.Markdown("Grades student answers using BERT-based semantic similarity and priority-weighted keyword matching.")

        with gr.Row():
            student_answer = gr.Textbox(label="Student Answer")
        reference_answers = gr.Textbox(label="Reference Answers (separate with ||)")
        
        # --- Integrated Keyword Extraction and Manual Edit ---
        def autofill_priority_keywords(ref_answers):
            ref_text = ref_answers.split('||')[0] if ref_answers else ""
            high, medium, low = extract_keywords_priority(ref_text)
            return (
                f"high: {', '.join(high)}\n"
                f"medium: {', '.join(medium)}\n"
                f"low: {', '.join(low)}"
            )

        with gr.Row():
            priority_keywords_input = gr.Textbox(
                label="Priority Keywords (Edit or use 'Extract from Reference')",
                lines=4,
                placeholder="high: ...\nmedium: ...\nlow: ..."
            )
            extract_btn = gr.Button("Extract from Reference", variant="secondary")

        extract_btn.click(
            autofill_priority_keywords,
            inputs=[reference_answers],
            outputs=priority_keywords_input
        )

        # --- Grade Answer at the bottom ---
        grade_btn = gr.Button("Grade Answer", variant="primary")
        grading_result = gr.Textbox(label="Grading Result", interactive=False, lines=8)

        # Only pass the required arguments (remove question/None)
        grade_btn.click(
            gradio_answer_grader,
            inputs=[student_answer, reference_answers, priority_keywords_input],
            outputs=grading_result
        )

    app.launch()