import gradio as gr
import os
import json
import re

# --- Import text grading logic ---
from AutogradTextModel import (
    extract_keywords_priority, clean_keyword, parse_priority_keywords,
    grade_answer, gradio_answer_grader
)

# --- Import diagram analysis logic ---
from AutogradImageModel import analyze_single_image

# Weight configurations for different grading types
TEXT_GRADING_WEIGHTS = [0.25, 0.55, 0.2]  # [bert_weight, sbert_weight, keyword_weight]
DIAGRAM_GRADING_WEIGHTS = [0.05, 0.95, 0.0]  # Stricter: SBERT dominates, BERT minor, no keyword

# Threshold configurations for different grading types
TEXT_GRADING_THRESHOLDS = {'A': 90, 'B': 80, 'C': 65, 'D': 25, 'F': 0}
DIAGRAM_GRADING_THRESHOLDS = {'A': 85, 'B': 80, 'C': 77, 'D': 75, 'F': 0}  # Even stricter: much higher cutoffs

# --- Helper: Keyword extraction for reference text/diagram ---
def extract_priority_keywords_from_text(ref_text):
    high, medium, low = extract_keywords_priority(ref_text)
    return (
        f"high: {', '.join(high)}\n"
        f"medium: {', '.join(medium)}\n"
        f"low: {', '.join(low)}"
    )

# --- Main combined grading logic ---
def combined_grader(
    student_text, student_diagram, diagram_type,
    reference_text, reference_diagram,
    priority_keywords_text # removed priority_keywords_diagram
):
    # --- Text grading ---
    text_grade_result = ""
    if student_text.strip() and reference_text.strip():
        # Split reference text by || to handle multiple versions
        ref_answers_list = [ans.strip() for ans in reference_text.split('||') if ans.strip()]
        priority_keywords = parse_priority_keywords(priority_keywords_text)
        grade = grade_answer(student_text, ref_answers_list, thresholds=TEXT_GRADING_THRESHOLDS, priority_keywords=priority_keywords, weights=TEXT_GRADING_WEIGHTS)
        text_grade_result = (
            f"**Text Answer Grading**\n"
            f"Semantic Similarity (BERT): {grade['semantic_similarity']:.4f}\n"
            f"Semantic Similarity (SBERT): {grade['sbert_similarity']:.4f}\n"
            f"Keyword Match Score: {grade['keyword_similarity']:.4f}\n"
            f"Weighted Keyword Match Percent: {grade['critical_match_percent']:.2f}%\n"
            f"Combined Score: {grade['combined_similarity']:.4f}\n"
            f"Scaled Score: {grade['bell_scaled_score']:.2f}\n"
            f"Grade: {grade['grade']}\n"
            f"Feedback: {grade['feedback']}\n"
        )
    elif student_text.strip():
        text_grade_result = "Student text answer provided, but no reference text for grading."
    elif reference_text.strip():
        text_grade_result = "Reference text provided, but no student text answer for grading."
    else:
        text_grade_result = "No text answer or reference provided."

    # --- Diagram grading ---
    diagram_grade_result = ""
    if student_diagram is not None and reference_diagram is not None:
        # Analyze student diagram once
        student_analysis = analyze_single_image(student_diagram, diagram_type)
        
        # Analyze reference diagram 5 times to get multiple versions
        reference_analyses = []
        for i in range(5):
            ref_analysis = analyze_single_image(reference_diagram, diagram_type)
            reference_analyses.append(ref_analysis)
        
        # Use all reference analyses for grading with diagram-specific weights and thresholds
        # No priority_keywords for diagram
        grade = grade_answer(student_analysis, reference_analyses, thresholds=DIAGRAM_GRADING_THRESHOLDS, priority_keywords=None, weights=DIAGRAM_GRADING_WEIGHTS)
        diagram_grade_result = (
            f"**Diagram Grading**\n"
            f"Semantic Similarity (BERT): {grade['semantic_similarity']:.4f}\n"
            f"Semantic Similarity (SBERT): {grade['sbert_similarity']:.4f}\n"
            f"Keyword Match Score: {grade['keyword_similarity']:.4f}\n"
            f"Weighted Keyword Match Percent: {grade['critical_match_percent']:.2f}%\n"
            f"Combined Score: {grade['combined_similarity']:.4f}\n"
            f"Scaled Score: {grade['bell_scaled_score']:.2f}\n"
            f"Grade: {grade['grade']}\n"
            f"Feedback: {grade['feedback']}\n"
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
    gr.Markdown("You can provide student text answer and/or diagram, and reference text and/or diagram. Extract keywords from reference, edit them, and grade both modalities.")

    with gr.Tab("Inputs"):
        with gr.Row():
            student_text = gr.Textbox(label="Student Text Answer", lines=4)
            student_diagram = gr.Image(type="filepath", label="Student Diagram")
        with gr.Row():
            reference_text = gr.Textbox(label="Reference Text", lines=4)
            reference_diagram = gr.Image(type="filepath", label="Reference Diagram")
        diagram_type = gr.Radio(
            choices=[
                "Flowchart", "Block Diagram", "Graph/Chart", "Table",
                "ER Diagram", "Network Diagram", "UML Diagram",
                "Circuit Diagram", "Scientific/Biological Diagram", "Mind Map/Concept Map"
            ],
            value="Flowchart",
            label="Diagram Type"
        )

    with gr.Tab("Keyword Extraction & Edit"):
        gr.Markdown("### Extract/Edit Priority Keywords for Text")
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
        # --- Remove diagram keyword extraction and manual editing UI ---

    gr.Markdown("## Grade")
    grade_btn = gr.Button("Grade Both")
    text_grade_result = gr.Textbox(label="Text Grading Result", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram, diagram_type,
            reference_text, reference_diagram,
            priority_keywords_text # removed priority_keywords_diagram
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )

if __name__ == "__main__":
    demo.launch()
