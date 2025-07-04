import gradio as gr
import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
import time  # Add time module for response time tracking

# Load the no-keywords grading model
no_keyword_model_path = os.path.join(os.path.dirname(__file__), "random_forest_grading_model_no_keywords.joblib")
if os.path.exists(no_keyword_model_path):
    rf_model_no_keywords = joblib.load(no_keyword_model_path)
    print("Loaded Random Forest grading model (no keywords)")
else:
    print(f"Warning: No-keyword model not found at {no_keyword_model_path}")
    rf_model_no_keywords = None

# Initialize NLP models for grading
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded NLP models for grading.")
except Exception as e:
    print(f"Error loading NLP models: {e}")
    tokenizer = bert_model = sbert_model = None

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
        thresholds = {'A': 80, 'B': 60, 'C': 40, 'D': 20, 'F': 0}
    
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

def diagram_grader(student_diagram, reference_diagram):
    """Main diagram grading function"""
    diagram_thresholds = {'A': 80, 'B': 60, 'C': 40, 'D': 20, 'F': 0}
    
    try:
        if student_diagram is not None and reference_diagram is not None:
            # Analyze both images together for comparison
            reference_desc, matching_sentences = analyze_both_images_for_comparison(student_diagram, reference_diagram)
            
            # Use the matching sentences for grading (this represents what the student got right)
            grade = grade_image_no_keywords(matching_sentences, [reference_desc], diagram_thresholds)
            
            # More aggressive truncation to prevent HTTP errors
            max_length = 200  # Reduced from 500
            ref_display = reference_desc[:max_length] + "..." if len(reference_desc) > max_length else reference_desc
            match_display = matching_sentences[:max_length] + "..." if len(matching_sentences) > max_length else matching_sentences
            
            # Ensure no control characters or problematic content
            ref_display = ref_display.replace('\n', ' ').replace('\r', ' ').strip()
            match_display = match_display.replace('\n', ' ').replace('\r', ' ').strip()
            
            diagram_grade_result = (
                f"Diagram Grading Results:\n"
                f"BERT Similarity: {grade['semantic_similarity']:.4f}\n"
                f"SBERT Similarity: {grade['sbert_similarity']:.4f}\n"
                f"Score: {grade['predicted_score']:.2f}/100\n"
                f"Grade: {grade['grade']}\n"
                f"Feedback: {grade['feedback']}\n\n"
                f"Reference: {ref_display}\n\n"
                f"Student Match: {match_display}"
            )
        elif student_diagram is not None:
            diagram_grade_result = "Student diagram provided, but no reference diagram for grading."
        elif reference_diagram is not None:
            diagram_grade_result = "Reference diagram provided, but no student diagram for grading."
        else:
            diagram_grade_result = "No diagram or reference provided."
    except Exception as e:
        diagram_grade_result = f"Error during grading: {str(e)[:100]}"

    # Final safety check - limit total response size
    if len(diagram_grade_result) > 1000:
        diagram_grade_result = diagram_grade_result[:1000] + "... (truncated)"

    return diagram_grade_result

# --- Gradio UI ---
with gr.Blocks(title="Diagram Grading Tool") as demo:
    gr.Markdown("# Diagram Grading Tool")
    gr.Markdown("**Upload student diagram and reference diagram for auto-grading using Random Forest model (no keywords).**")

    with gr.Tab("Diagram Grading"):
        with gr.Row():
            student_diagram = gr.Image(type="filepath", label="Student Diagram")
            reference_diagram = gr.Image(type="filepath", label="Reference Diagram")

    gr.Markdown("## Grade")
    grade_btn = gr.Button("Grade Diagram", variant="primary")
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result", interactive=False, lines=12)

    grade_btn.click(
        diagram_grader,
        inputs=[student_diagram, reference_diagram],
        outputs=[diagram_grade_result]
    )

if __name__ == "__main__":
    demo.launch()