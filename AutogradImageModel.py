import gradio as gr
import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat

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
    """Analyze both images together using LLaVA to find common elements."""
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

    # Comparison prompt for finding matching elements
    comparison_prompt = (
        "You are given two diagrams: the first is a student's answer and the second is the reference/correct answer.\n\n"
        "TASK 1 - DESCRIBE REFERENCE IMAGE:\n"
        "Look at the second image (reference/teacher's answer) and describe it completely.\n"
        "Write each unique fact as a separate sentence. Avoid repeating the same information.\n"
        "Example format:\n"
        "- The diagram has a start oval labeled 'Begin'.\n"
        "- There is a decision diamond labeled 'x > 5?'.\n"
        "- The start oval connects to the decision diamond with an arrow.\n"
        "- The decision diamond has two outputs: 'Yes' and 'No'.\n\n"
        "TASK 2 - VERIFY EACH SENTENCE INDIVIDUALLY:\n"
        "Now look at the first image (student's answer). For EACH INDIVIDUAL sentence you wrote about the reference, "
        "carefully examine the student's image and ask these specific questions:\n\n"
        "For each sentence, check:\n"
        "1. Does this exact component exist in the student image?\n"
        "2. Is the shape type correct (oval, rectangle, diamond, etc.)?\n"
        "3. Is the text content at least 50% similar?\n"
        "4. If it's a connection, does this connection actually exist?\n"
        "5. Are the spatial relationships accurate?\n\n"
        "STRICT VERIFICATION RULES:\n"
        "- If ANY major element of the sentence is wrong, DO NOT include it\n"
        "- If the component doesn't exist at all in student image, DO NOT include it\n"
        "- If the shape is completely different (oval vs rectangle), DO NOT include it\n"
        "- If the text is completely different, DO NOT include it\n"
        "- If the connection doesn't exist, DO NOT include it\n"
        "- Only include if the sentence is factually accurate about the student's diagram\n\n"
        "Go through each sentence one by one and verify it individually against the student image.\n\n"
        "Format your response exactly as:\n"
        "REFERENCE DESCRIPTION:\n"
        "[Each unique sentence on a new line]\n\n"
        "MATCHING SENTENCES:\n"
        "[Only sentences that are factually accurate about the student's image after individual verification]"
    )
    
    response = chat(
        model='llava',
        messages=[{'role': 'user', 'content': comparison_prompt, 'images': [student_bytes, reference_bytes]}]
    )
    
    full_analysis = response['message']['content']
    
    # Extract the two sections
    lines = full_analysis.split('\n')
    reference_desc = ""
    student_matching = ""
    current_section = None
    
    ref_sentences = set()
    matching_sentences = set()
    
    for line in lines:
        line = line.strip()
        if "REFERENCE DESCRIPTION:" in line:
            current_section = "ref"
            continue
        elif "MATCHING SENTENCES:" in line:
            current_section = "student"
            continue
        elif current_section == "ref" and line and line not in ref_sentences:
            ref_sentences.add(line)
            reference_desc += line + "\n"
        elif current_section == "student" and line and line not in matching_sentences:
            matching_sentences.add(line)
            student_matching += line + "\n"
    
    return reference_desc.strip(), student_matching.strip()

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