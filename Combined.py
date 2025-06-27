import gradio as gr
import os
import json
import re
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
import nltk
from nltk.corpus import stopwords
import yake
from keybert import KeyBERT
import time  # Add time module for response time tracking

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load both models
keyword_model_path = os.path.join(os.path.dirname(__file__), "random_forest_grading_model.joblib")
no_keyword_model_path = os.path.join(os.path.dirname(__file__), "random_forest_grading_model_no_keywords.joblib")

if os.path.exists(keyword_model_path):
    rf_model_keywords = joblib.load(keyword_model_path)
    print("Loaded Random Forest grading model (with keywords)")
else:
    print(f"Warning: Keyword model not found at {keyword_model_path}")
    rf_model_keywords = None

if os.path.exists(no_keyword_model_path):
    rf_model_no_keywords = joblib.load(no_keyword_model_path)
    print("Loaded Random Forest grading model (no keywords)")
else:
    print(f"Warning: No-keyword model not found at {no_keyword_model_path}")
    rf_model_no_keywords = None

# Load common words for keyword extraction
COMMON_WORDS = set()
common_words_path = os.path.join(os.path.dirname(__file__), "common_words.json")
if os.path.exists(common_words_path):
    with open(common_words_path, "r", encoding="utf-8") as f:
        COMMON_WORDS = set(json.load(f))
COMMON_WORDS = COMMON_WORDS.union(set(stopwords.words('english')))

# Initialize NLP models
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT()
    print("Loaded NLP models.")
except Exception as e:
    print(f"Error loading NLP models: {e}")
    tokenizer = bert_model = sbert_model = kw_model = None

# Keyword extraction functions
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
    if not kw_model:
        return [], [], []
    
    keybert_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=12,
        top_n=8
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
    high = [k for k, _ in sorted_keywords[:high_n]]
    medium = [k for k, _ in sorted_keywords[high_n:high_n+med_n]]
    low = [k for k, _ in sorted_keywords[high_n+med_n:]]
    return high, medium, low

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

# Load the trained Random Forest model (no keywords version)
model_path = os.path.join(os.path.dirname(__file__), "random_forest_grading_model_no_keywords.joblib")
if os.path.exists(model_path):
    rf_model = joblib.load(model_path)
    print("Loaded Random Forest grading model (no keywords)")
else:
    print(f"Warning: Model not found at {model_path}")
    rf_model = None

# Initialize NLP models
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded NLP models.")
except Exception as e:
    print(f"Error loading NLP models: {e}")
    tokenizer = bert_model = sbert_model = None

def analyze_single_image(image):
    """Analyze diagram image using LLaVA with strict deterministic prompt."""
    if image is None:
        return "Please provide an image."

    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            img_bytes = img_file.read()
    else:
        img_bytes = image.read()

    # Highly deterministic prompt with numbered labeling system
    strict_prompt = (
        "Analyze this diagram and provide a systematic description using numbered labels:\n\n"
        "STEP 1 - COMPONENT LABELING:\n"
        "Assign a number (1, 2, 3, etc.) to each distinct box, shape, symbol, or component in the diagram. "
        "Go from left-to-right, top-to-bottom order. For each component, provide:\n"
        "- Component [NUMBER]: [SHAPE TYPE] containing text '[EXACT TEXT]'\n\n"
        "STEP 2 - CONNECTION MAPPING:\n"
        "List all connections using only the assigned numbers:\n"
        "- [NUMBER] connects to [NUMBER] (describe arrow direction/line type)\n\n"
        "STEP 3 - FLOW SEQUENCE:\n"
        "Describe the logical flow using only component numbers:\n"
        "Start: [NUMBER] → [NUMBER] → [NUMBER] → End: [NUMBER]\n\n"
        "STEP 4 - TEXT INVENTORY:\n"
        "List all visible text exactly as written:\n"
        "- Text in component [NUMBER]: '[EXACT TEXT]'\n\n"
        "Be extremely precise with numbering and text transcription. Use consistent terminology."
    )
    
    response = chat(
        model='llava',
        messages=[{'role': 'user', 'content': strict_prompt, 'images': [img_bytes]}]
    )
    
    return response['message']['content']

def get_bert_embedding(text):
    """Get BERT embedding for a given text."""
    if not tokenizer or not bert_model:
        return None
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def calculate_similarity(student_answer, reference_answer):
    """Calculate BERT-based similarity between two texts."""
    embedding1 = get_bert_embedding(student_answer)
    embedding2 = get_bert_embedding(reference_answer)
    if embedding1 is None or embedding2 is None:
        return 0.0
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

def calculate_sbert_similarity(student_answer, reference_answer):
    """Calculate SBERT-based similarity between two texts."""
    if not sbert_model:
        return 0.0
    emb1 = sbert_model.encode([student_answer])[0]
    emb2 = sbert_model.encode([reference_answer])[0]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return sim

def predict_score_with_keywords(bert_similarity, sbert_similarity, keyword_similarity):
    """Predict score using the trained Random Forest model (with keywords)"""
    if rf_model_keywords is None:
        return 50.0  # Default score if model not loaded
    
    features = np.array([[bert_similarity, sbert_similarity, keyword_similarity]])
    predicted_score = rf_model_keywords.predict(features)[0]
    
    # Convert from 0-1 range to 0-100 range
    predicted_score = predicted_score * 100
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, predicted_score))

def predict_score_no_keywords(bert_similarity, sbert_similarity):
    """Predict score using the trained Random Forest model (no keywords)"""
    if rf_model_no_keywords is None:
        return 50.0  # Default score if model not loaded
    
    features = np.array([[bert_similarity, sbert_similarity]])
    predicted_score = rf_model_no_keywords.predict(features)[0]
    
    # Convert from 0-1 range to 0-100 range
    predicted_score = predicted_score * 100
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, predicted_score))

def score_to_grade(score, thresholds=None):
    """Convert numerical score (0-100) to letter grade"""
    if thresholds is None:
        thresholds = {'A': 90, 'B': 80, 'C': 65, 'D': 25, 'F': 0}
    
    if score >= thresholds['A']:
        return 'A'
    elif score >= thresholds['B']:
        return 'B'
    elif score >= thresholds['C']:
        return 'C'
    elif score >= thresholds['D']:
        return 'D'
    else:
        return 'F'

def generate_feedback(score, grade):
    """Generate feedback based on score and grade"""
    feedback_map = {
        'A': "Excellent answer! Your response demonstrates strong understanding.",
        'B': "Good answer. Your response captures most key concepts well.",
        'C': "Acceptable answer. Consider adding more details or key concepts.",
        'D': "Below average answer. Review the material and try to include more relevant information.",
        'F': "Your answer needs significant improvement. Please review the material thoroughly."
    }
    return feedback_map.get(grade, "Unable to generate feedback.")

def grade_text_with_keywords(student_answer, reference_answers, priority_keywords=None, thresholds=None):
    """Grade text answer using Random Forest model with keywords"""
    # Calculate similarities
    best_bert_similarity = max(calculate_similarity(student_answer, ref) for ref in reference_answers)
    best_sbert_similarity = max(calculate_sbert_similarity(student_answer, ref) for ref in reference_answers)
    
    keyword_similarity, missing_keywords, critical_missing, critical_match_percent = calculate_keyword_similarity(
        student_answer, reference_answers, priority_keywords
    )
    
    # Predict score using Random Forest with keywords
    predicted_score = predict_score_with_keywords(best_bert_similarity, best_sbert_similarity, keyword_similarity)
    
    # Convert to grade
    grade = score_to_grade(predicted_score, thresholds)

    
    return {
        'semantic_similarity': best_bert_similarity,
        'sbert_similarity': best_sbert_similarity,
        'keyword_similarity': keyword_similarity,
        'predicted_score': predicted_score,
        'grade': grade,
        'missing_keywords': missing_keywords,
        'critical_missing': critical_missing,
        'critical_match_percent': critical_match_percent,
        'feedback': generate_feedback(predicted_score, grade)
    }

def grade_image_no_keywords(student_answer, reference_answers, thresholds=None):
    """Grade image answer using Random Forest model without keywords"""
    # Calculate similarities
    best_bert_similarity = max(calculate_similarity(student_answer, ref) for ref in reference_answers)
    best_sbert_similarity = max(calculate_sbert_similarity(student_answer, ref) for ref in reference_answers)
    
    # Always use Random Forest model for prediction
    predicted_score = predict_score_no_keywords(best_bert_similarity, best_sbert_similarity)
    
    # Ensure score is between 0 and 100
    predicted_score = max(0.0, min(100.0, predicted_score))
    
    # Convert to grade
    grade = score_to_grade(predicted_score, thresholds)
    
    return {
        'semantic_similarity': best_bert_similarity,
        'sbert_similarity': best_sbert_similarity,
        'predicted_score': predicted_score,
        'grade': grade,
        'feedback': generate_feedback(predicted_score, grade)
    }

def extract_priority_keywords_from_text(ref_text):
    """Extract priority keywords from reference text"""
    if not ref_text.strip():
        return ""
    high, medium, low = extract_keywords_priority(ref_text)
    return (
        f"high: {', '.join(high)}\n"
        f"medium: {', '.join(medium)}\n"
        f"low: {', '.join(low)}"
    )

def parse_priority_keywords(priority_keywords_text):
    """Parse priority keywords from text format"""
    priority_keywords = {'high': [], 'medium': [], 'low': []}
    
    if not priority_keywords_text.strip():
        return priority_keywords
    
    lines = priority_keywords_text.strip().split('\n')
    for line in lines:
        if ':' in line:
            priority, keywords = line.split(':', 1)
            priority = priority.strip().lower()
            if priority in priority_keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                priority_keywords[priority] = keywords_list
    
    return priority_keywords

def analyze_both_images_for_comparison(student_image, reference_image):
    """Analyze both images separately using a strict, numbered format prompt."""
    if student_image is None or reference_image is None:
        return "Please provide both images.", ""

    # Load both images
    if isinstance(student_image, str):
        with open(student_image, 'rb') as img_file:
            student_bytes = img_file.read()
    else:
        student_bytes = student_image.read()
        
    if isinstance(reference_image, str):
        with open(reference_image, 'rb') as img_file:
            reference_bytes = img_file.read()
    else:
        reference_bytes = reference_image.read()

    # Strict, numbered format prompt for analyzing diagrams
    strict_prompt = (
        "Explain the following diagram in detail, focusing on the key elements and their relationships. "
    )

    # Analyze the reference image
    start_time = time.time()  # Start timing
    try:
        reference_response = chat(
            model='llava',
            messages=[{'role': 'user', 'content': strict_prompt, 'images': [reference_bytes]}],
            options={
                'temperature': 0.0,      # No randomness in sampling
                'top_p': 0.0,            # No nucleus sampling
                'top_k': 1,              # Greedy decoding
                'do_sample': False,      # Use deterministic decoding
            }
        )
        reference_description = reference_response['message']['content']
    except Exception as e:
        reference_description = "Error analyzing reference image."

    # Analyze the student image
    start_time = time.time()  # Start timing
    try:
        student_response = chat(
            model='llava',
            messages=[{'role': 'user', 'content': strict_prompt, 'images': [student_bytes]}],
            options={
                'temperature': 0.0,      # No randomness in sampling
                'top_p': 0.0,            # No nucleus sampling
                'top_k': 1,              # Greedy decoding
                'do_sample': False,      # Use deterministic decoding
            }
        )
        student_description = student_response['message']['content']
    except Exception as e:
        student_description = "Error analyzing student image."

    return reference_description.strip(), student_description.strip()

# --- Main combined grading logic ---
def combined_grader(
    student_text, student_diagram,
    reference_text, reference_diagram, priority_keywords_text
):
    # Text grading thresholds
    text_thresholds = {'A': 90, 'B': 80, 'C': 65, 'D': 25, 'F': 0}
    # Diagram grading thresholds (stricter)
    diagram_thresholds = {'A': 80, 'B': 60, 'C': 40, 'D': 20, 'F': 0}
    
    # --- Text grading (with keywords) ---
    text_grade_result = ""
    if student_text.strip() and reference_text.strip():
        ref_answers_list = [ans.strip() for ans in reference_text.split('||') if ans.strip()]
        priority_keywords = parse_priority_keywords(priority_keywords_text)
        
        grade = grade_text_with_keywords(student_text, ref_answers_list, priority_keywords, text_thresholds)
        text_grade_result = (
            f"**Text Answer Grading (Random Forest Model - With Keywords)**\n"
            f"Semantic Similarity (BERT): {grade['semantic_similarity']:.4f}\n"
            f"Semantic Similarity (SBERT): {grade['sbert_similarity']:.4f}\n"
            f"Keyword Match Score: {grade['keyword_similarity']:.4f}\n"
            f"Weighted Keyword Match Percent: {grade['critical_match_percent']:.2f}%\n"
            f"Predicted Score: {grade['predicted_score']:.2f}/100\n"
            f"Grade: {grade['grade']}\n"
            f"Feedback: {grade['feedback']}\n"
        )
    elif student_text.strip():
        text_grade_result = "Student text answer provided, but no reference text for grading."
    elif reference_text.strip():
        text_grade_result = "Reference text provided, but no student text answer for grading."
    else:
        text_grade_result = "No text answer or reference provided."

    # --- Diagram grading (no keywords) using comparison analysis ---
    diagram_grade_result = ""
    if student_diagram is not None and reference_diagram is not None:
        # Analyze both images together for comparison
        reference_desc, matching_sentences = analyze_both_images_for_comparison(student_diagram, reference_diagram)
        
        # Use the matching sentences for grading (this represents what the student got right)
        grade = grade_image_no_keywords(matching_sentences, [reference_desc], diagram_thresholds)
        
        # Truncate long descriptions to prevent HTTP errors
        ref_display = reference_desc[:500] + "..." if len(reference_desc) > 500 else reference_desc
        match_display = matching_sentences[:500] + "..." if len(matching_sentences) > 500 else matching_sentences
        
        diagram_grade_result = (
            f"**Diagram Grading (Random Forest Model - No Keywords)**\n"
            f"Semantic Similarity (BERT): {grade['semantic_similarity']:.4f}\n"
            f"Semantic Similarity (SBERT): {grade['sbert_similarity']:.4f}\n"
            f"Predicted Score: {grade['predicted_score']:.2f}/100\n"
            f"Grade: {grade['grade']}\n"
            f"Feedback: {grade['feedback']}\n\n"
            f"**Reference Answer (Complete Description):**\n{ref_display}\n\n"
            f"**What Student Got Correct (Matching Elements):**\n{match_display}"
        )
    elif student_diagram is not None:
        diagram_grade_result = "Student diagram provided, but no reference diagram for grading."
    elif reference_diagram is not None:
        diagram_grade_result = "Reference diagram provided, but no student diagram for grading."
    else:
        diagram_grade_result = "No diagram or reference provided."

    return text_grade_result, diagram_grade_result

# --- Gradio UI ---
with gr.Blocks(title="Combined Text & Diagram Grading Tool") as demo:
    gr.Markdown("# Combined Text & Diagram Grading Tool")
    gr.Markdown("**Text grading uses keywords, Image grading uses no keywords!** Provide student text answer and/or diagram, and reference text and/or diagram.")

    with gr.Tab("Inputs"):
        with gr.Row():
            student_text = gr.Textbox(label="Student Text Answer", lines=4)
            student_diagram = gr.Image(type="filepath", label="Student Diagram")
        with gr.Row():
            reference_text = gr.Textbox(label="Reference Text", lines=4)
            reference_diagram = gr.Image(type="filepath", label="Reference Diagram")

    with gr.Tab("Keyword Extraction & Edit"):
        gr.Markdown("### Extract/Edit Priority Keywords for Text Grading")
        with gr.Row():
            priority_keywords_text = gr.Textbox(
                label="Priority Keywords for Text (edit or extract)",
                lines=4,
                placeholder="high: ...\nmedium: ...\nlow: ..."
            )
            extract_kw_text_btn = gr.Button("Extract from Reference Text", variant="secondary")
        extract_kw_text_btn.click(
            extract_priority_keywords_from_text,
            inputs=[reference_text],
            outputs=priority_keywords_text
        )

    gr.Markdown("## Grade")
    grade_btn = gr.Button("Grade Both", variant="primary")
    text_grade_result = gr.Textbox(label="Text Grading Result (With Keywords)", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result (No Keywords)", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram,
            reference_text, reference_diagram, priority_keywords_text
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )

if __name__ == "__main__":
    demo.launch()
    grade_btn = gr.Button("Grade Both", variant="primary")
    text_grade_result = gr.Textbox(label="Text Grading Result (With Keywords)", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result (No Keywords)", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram,
            reference_text, reference_diagram, priority_keywords_text
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )
    text_grade_result = gr.Textbox(label="Text Grading Result (With Keywords)", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result (No Keywords)", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram,
            reference_text, reference_diagram, priority_keywords_text
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )

if __name__ == "__main__":
    demo.launch()
    grade_btn = gr.Button("Grade Both", variant="primary")
    text_grade_result = gr.Textbox(label="Text Grading Result (With Keywords)", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result (No Keywords)", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram,
            reference_text, reference_diagram, priority_keywords_text
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )

