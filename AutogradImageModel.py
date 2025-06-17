import gradio as gr
from ollama import chat
import os
from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
import string
import re

# Output folder
os.makedirs("outputs", exist_ok=True)

# Function to analyze each image
def analyze_single_image(image, diagram_type):
    if image is None:
        return "Please provide an image."

    # If image is a file path, open and read as bytes
    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            img_bytes = img_file.read()
    else:
         img_bytes = image.read()

    if diagram_type == "Flowchart":
     adjusted_prompt = (
        f"Examine the flowchart in this image and provide a single-paragraph explanation "
        f"that identifies and describes every labeled process, symbol, decision point, and arrow. "
        f"Detail the step-by-step flow of the logic or operation, including any loops or conditions. "
        f"Accurately describe the flow direction, starting and ending points, and hierarchical structure "
        f"as depicted, ensuring that all components are included in the output."
    )
    elif diagram_type == "Block Diagram":
        adjusted_prompt = (
            f"Analyze the block diagram shown in this image. Describe in a single paragraph the function "
            f"of each labeled block or module, the connections between them, and the directional flow of data or control. "
            f"Explain the layout structure, hierarchy, and how each component interacts, ensuring all names, arrows, "
            f"and symbols are transcribed precisely and organized clearly for comparison."
        )
    elif diagram_type == "Graph/Chart":
        adjusted_prompt = (
            f"Carefully examine the graph or chart in this image and provide a detailed single-paragraph description. "
            f"Identify and transcribe all axis labels, titles, legends, and data values. For each bar/line/segment, specify "
            f"its value and its associated category. Describe any trends, relationships, or comparisons depicted, and explain "
            f"the chart's organization to enable accurate reproduction and evaluation."
        )
    elif diagram_type == "Table":
        adjusted_prompt = (
            f"Read the table in the image and generate a single-paragraph summary that transcribes the title, each header, "
            f"and the content of every cell. For each data point, specify the corresponding row and column it belongs to. "
            f"Ensure the structure and organization of the table is described objectively and precisely, preserving its hierarchy "
            f"and labeling."
        )
    elif diagram_type == "ER Diagram":
        adjusted_prompt = (
            f"Examine the ER diagram and describe in a single paragraph all entities, attributes, and relationships. "
            f"Identify primary/foreign keys, cardinalities, and connections. Specify the nature of relationships "
            f"(one-to-many, many-to-many, etc.) and label each component as it appears, maintaining logical and spatial ordering "
            f"for exact comparison."
        )
    elif diagram_type == "Network Diagram":
        adjusted_prompt = (
            f"Analyze the network diagram provided. Describe each labeled component such as nodes, devices, or layers. "
            f"Explain the roles of each part, their connections, and the direction of data flow. Clearly interpret stack structures "
            f"or hierarchies (e.g., OSI layers from top to bottom), providing one detailed paragraph that preserves positional "
            f"and logical meaning."
        )
    elif diagram_type == "UML Diagram":
        adjusted_prompt = (
            f"Interpret the UML diagram in the image and describe in a single paragraph all classes, objects, methods, "
            f"attributes, and relationships. Include visibility indicators (+/-/#), inheritance, associations, and multiplicities. "
            f"Accurately represent each component's position, label, and connections to maintain the diagram's semantic structure."
        )
    elif diagram_type == "Circuit Diagram":
        adjusted_prompt = (
            f"Describe the circuit diagram in this image with a detailed single-paragraph explanation. Identify each component "
            f"(resistors, capacitors, power sources, etc.), their values or labels, and explain the connections between them. "
            f"Include direction of current flow, input/output terminals, and overall configuration clearly for objective interpretation."
        )
    elif diagram_type == "Scientific/Biological Diagram":
        adjusted_prompt = (
            f"Carefully analyze the labeled diagram and provide a single-paragraph explanation that describes each part, label, "
            f"and its function. Maintain the layout order, direction, and grouping of components. Clearly state how each part relates "
            f"to others within the system or structure, ensuring exact transcription and comprehensive description."
        )
    elif diagram_type == "Mind Map/Concept Map":
        adjusted_prompt = (
            f"Interpret the concept map in this image and generate a detailed single-paragraph explanation. Identify the central "
            f"idea and each branching node, including their labels and the relationships or flows between them. Explain the hierarchical "
            f"or logical order of concepts, ensuring all nodes and connections are covered systematically."
        )
    else:
     adjusted_prompt = "Please select a diagram type."

    response = chat(
    model='llava',
    messages=[{'role': 'user', 'content': adjusted_prompt, 'images': [img_bytes]}]
)

    return response['message']['content']

# Main function to handle both student and reference image
def process_both_images(student_img, reference_img, diagram_type):
    student_answer = analyze_single_image(student_img, diagram_type)
    reference_answer = analyze_single_image(reference_img, diagram_type)
    combined = f"🧑‍🎓 **Student Answer:**\n{student_answer}\n\n📘 **Reference Answer:**\n{reference_answer}"
    return student_answer, reference_answer, combined

# Import actual models
from keybert import KeyBERT
import yake

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load common words from JSON and combine with stopwords
with open(os.path.join(os.path.dirname(__file__), "common_words.json"), "r", encoding="utf-8") as f:
    COMMON_WORDS = set(json.load(f))
COMMON_WORDS = COMMON_WORDS.union(set(stopwords.words('english')))

kw_model = KeyBERT()
yake_extractor = yake.KeywordExtractor()

# Reduce nr_candidates and top_n for faster extraction
KEYBERT_TOP_N = 8
YAKE_TOP_N = 8

def clean_keyword(keyword):
    """Clean and validate a single keyword"""
    # Remove punctuation and convert to lowercase
    keyword = re.sub(r'[^\w\s]', '', keyword.lower().strip())
    
    # Skip if empty, too short, or common word
    if len(keyword) < 3 or keyword in COMMON_WORDS:
        return None
    
    # Skip if it's all digits
    if keyword.isdigit():
        return None
        
    return keyword

def extract_keywords_priority(text):
    """
    Extracts keywords and categorizes them into high, medium, and low priority
    based on KeyBERT scores (descending).
    """
    # KeyBERT - extract single words only, fewer candidates for speed
    keybert_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=12,
        top_n=KEYBERT_TOP_N
    )
    # keybert_keywords: list of (keyword, score)
    # Clean and filter keywords, deduplicate early
    cleaned_keywords = []
    seen = set()
    for keyword, score in keybert_keywords:
        cleaned = clean_keyword(keyword)
        if cleaned and cleaned not in seen:
            cleaned_keywords.append((cleaned, score))
            seen.add(cleaned)

    # Sort by score descending
    sorted_keywords = sorted(cleaned_keywords, key=lambda x: -x[1])
    total = len(sorted_keywords)
    if total == 0:
        return [], [], []

    # Assign high, medium, low by percentage
    high_n = max(1, int(total * 0.3))
    med_n = max(1, int(total * 0.4))
    low_n = total - high_n - med_n

    high = [k for k, _ in sorted_keywords[:high_n]]
    medium = [k for k, _ in sorted_keywords[high_n:high_n+med_n]]
    low = [k for k, _ in sorted_keywords[high_n+med_n:]]

    return high, medium, low

# Store the last auto-generated keywords for merging with manual keywords
last_auto_keywords = {"high": [], "medium": [], "low": []}

def generate_auto_keywords(reference_answer):
    """Generate keywords automatically from reference answer only"""
    global last_auto_keywords
    if not reference_answer:
        last_auto_keywords = {"high": [], "medium": [], "low": []}
        return "Please analyze images first to generate keywords."
    
    ref_high, ref_med, ref_low = extract_keywords_priority(reference_answer)
    last_auto_keywords = {
        "high": ref_high,
        "medium": ref_med,
        "low": ref_low
    }
    result = (
        f"**Reference Keywords:**\n"
        f"High Priority: {', '.join(ref_high)}\n"
        f"Medium Priority: {', '.join(ref_med)}\n"
        f"Low Priority: {', '.join(ref_low)}"
    )
    return result

def process_manual_keywords(manual_keywords_text):
    """
    Process manually entered keywords and merge with auto-generated keywords.
    Display all in three sections: High, Medium, Low.
    """
    global last_auto_keywords
    if not manual_keywords_text.strip():
        # Show current auto keywords if no manual input
        return (
            f"High Priority: {', '.join(last_auto_keywords['high'])}\n"
            f"Medium Priority: {', '.join(last_auto_keywords['medium'])}\n"
            f"Low Priority: {', '.join(last_auto_keywords['low'])}"
        )
    
    # Split by comma and clean each keyword
    manual_keywords = [kw.strip() for kw in manual_keywords_text.split(',')]
    cleaned_keywords = []
    for keyword in manual_keywords:
        clean_kw = clean_keyword(keyword)
        if clean_kw:
            cleaned_keywords.append(clean_kw)
    if not cleaned_keywords:
        return "No valid keywords found after filtering."

    # Merge manual keywords into auto keywords, distributing evenly
    # Try to add new manual keywords to high, then medium, then low, in round-robin fashion
    merged = {
        "high": list(last_auto_keywords["high"]),
        "medium": list(last_auto_keywords["medium"]),
        "low": list(last_auto_keywords["low"])
    }
    # Flatten all auto keywords for duplicate checking
    all_auto = set(merged["high"] + merged["medium"] + merged["low"])
    # Distribute manual keywords
    sections = ["high", "medium", "low"]
    idx = 0
    for kw in cleaned_keywords:
        if kw not in all_auto:
            merged[sections[idx % 3]].append(kw)
            idx += 1

    result = (
        f"High Priority: {', '.join(merged['high'])}\n"
        f"Medium Priority: {', '.join(merged['medium'])}\n"
        f"Low Priority: {', '.join(merged['low'])}"
    )
    # Update last_auto_keywords so further manual adds are cumulative
    last_auto_keywords = merged
    return result

# Define the main_process function
def main_process(student_img, reference_img, diagram_type):
    s_ans, r_ans, _ = process_both_images(student_img, reference_img, diagram_type)
    return s_ans, r_ans

# Gradio UI
with gr.Blocks(title="Diagram Analysis Tool") as demo:
    # Inline CSS for Gradio UI
    gr.HTML("""
    <style>
    /* Example custom styles, adjust as needed */
    .gr-button { font-weight: bold; }
    .gr-textbox textarea { font-size: 1.05em; }
    .gr-markdown h1, .gr-markdown h2 { color: #2d3748; }
    </style>
    """)
    gr.Markdown("# Compare Student and Reference Diagrams")
    gr.Markdown("Upload both student and reference diagram images. Select the diagram type for accurate analysis and comparison.")

    with gr.Row():
        student_img = gr.Image(type="filepath", label="Upload Student Image")
        reference_img = gr.Image(type="filepath", label="Upload Reference Image")
    
    diagram_type = gr.Radio(
        choices=[
            "Flowchart", "Block Diagram", "Graph/Chart", "Table",
            "ER Diagram", "Network Diagram", "UML Diagram",
            "Circuit Diagram", "Scientific/Biological Diagram", "Mind Map/Concept Map"
        ],
        value="Flowchart",
        label="Select Diagram Type"
    )
    
    # Keywords section
    gr.Markdown("## Keywords Extraction")
    
    analyze_btn = gr.Button("Analyze Images", variant="primary")

    with gr.Row():
        student_answer = gr.Textbox(label="Student Answer", interactive=False, lines=5)
        reference_answer = gr.Textbox(label="Reference Answer", interactive=False, lines=5)

    auto_keywords_btn = gr.Button("Generate Keywords Automatically", variant="secondary")

    with gr.Row():
        manual_keywords_btn = gr.Button("Add Keywords Manually", variant="secondary")

    # --- Manual keyword input: 3 boxes for high, medium, low ---
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
    keywords_output = gr.Textbox(label="Keywords Output", interactive=False, lines=6)

    # --- Event handlers ---
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

    def process_manual_keywords_v2(high_text, medium_text, low_text):
        global last_auto_keywords
        # Parse and clean each section
        def parse_section(text):
            return [clean_keyword(kw) for kw in text.split(",") if clean_keyword(kw)]
        manual_high = parse_section(high_text)
        manual_medium = parse_section(medium_text)
        manual_low = parse_section(low_text)

        # Merge with auto keywords, only add if not present in that section
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

    # Connect events
    analyze_btn.click(
        main_process,
        inputs=[student_img, reference_img, diagram_type],
        outputs=[student_answer, reference_answer]
    )

    auto_keywords_btn.click(
        generate_auto_keywords,
        inputs=[reference_answer],
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
    demo.launch()