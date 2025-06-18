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

# --- Helper: Keyword extraction for reference text/diagram ---
def extract_priority_keywords_from_text(ref_text):
    high, medium, low = extract_keywords_priority(ref_text)
    return (
        f"high: {', '.join(high)}\n"
        f"medium: {', '.join(medium)}\n"
        f"low: {', '.join(low)}"
    )

def extract_priority_keywords_from_diagram(diagram_img, diagram_type):
    if diagram_img is None:
        return ""
    diagram_analysis = analyze_single_image(diagram_img, diagram_type)
    return extract_priority_keywords_from_text(diagram_analysis)

# --- Main combined grading logic ---
def combined_grader(
    student_text, student_diagram, diagram_type,
    reference_text, reference_diagram,
    priority_keywords_text, priority_keywords_diagram
):
    # --- Text grading ---
    text_grade_result = ""
    if student_text.strip() and reference_text.strip():
        # Use text grading logic
        ref_answers_list = [reference_text.strip()]
        priority_keywords = parse_priority_keywords(priority_keywords_text)
        grade = grade_answer(student_text, ref_answers_list, priority_keywords=priority_keywords)
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
        # Analyze both diagrams
        student_analysis = analyze_single_image(student_diagram, diagram_type)
        reference_analysis = analyze_single_image(reference_diagram, diagram_type)
        # Use text grading logic on the analyses
        ref_answers_list = [reference_analysis]
        priority_keywords = parse_priority_keywords(priority_keywords_diagram)
        grade = grade_answer(student_analysis, ref_answers_list, priority_keywords=priority_keywords)
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

        gr.Markdown("### Extract/Edit Priority Keywords for Diagram")
        with gr.Row():
            priority_keywords_diagram = gr.Textbox(
                label="Priority Keywords for Diagram (edit or extract)",
                lines=4,
                placeholder="high: ...\nmedium: ...\nlow: ..."
            )
            extract_kw_diagram_btn = gr.Button("Extract from Reference Diagram", variant="secondary")
        extract_kw_diagram_btn.click(
            extract_priority_keywords_from_diagram,
            inputs=[reference_diagram, diagram_type],
            outputs=priority_keywords_diagram
        )

    gr.Markdown("## Grade")
    grade_btn = gr.Button("Grade Both")
    text_grade_result = gr.Textbox(label="Text Grading Result", interactive=False, lines=8)
    diagram_grade_result = gr.Textbox(label="Diagram Grading Result", interactive=False, lines=8)

    grade_btn.click(
        combined_grader,
        inputs=[
            student_text, student_diagram, diagram_type,
            reference_text, reference_diagram,
            priority_keywords_text, priority_keywords_diagram
        ],
        outputs=[text_grade_result, diagram_grade_result]
    )

if __name__ == "__main__":
    demo.launch()
