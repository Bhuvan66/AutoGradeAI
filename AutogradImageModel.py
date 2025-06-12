import gradio as gr
from ollama import chat
import os

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

    # Prompt generator
    prompt_map = {
        "Flowchart": "Examine the flowchart in this image and provide a single-paragraph explanation that identifies and describes every labeled process, symbol, decision point, and arrow. Detail the step-by-step flow...",
        "Block Diagram": "Analyze the block diagram shown in this image. Describe in a single paragraph the function of each labeled block or module...",
        "Graph/Chart": "Carefully examine the graph or chart in this image and provide a detailed single-paragraph description...",
        "Table": "Read the table in the image and generate a single-paragraph summary that transcribes the title, each header, and the content of every cell...",
        "ER Diagram": "Examine the ER diagram and describe in a single paragraph all entities, attributes, and relationships...",
        "Network Diagram": "Analyze the network diagram provided. Describe each labeled component such as nodes, devices, or layers...",
        "UML Diagram": "Interpret the UML diagram in the image and describe in a single paragraph all classes, objects, methods, attributes, and relationships...",
        "Circuit Diagram": "Describe the circuit diagram in this image with a detailed single-paragraph explanation...",
        "Scientific/Biological Diagram": "Carefully analyze the labeled diagram and provide a single-paragraph explanation that describes each part...",
        "Mind Map/Concept Map": "Interpret the concept map in this image and generate a detailed single-paragraph explanation..."
    }

    prompt = prompt_map.get(diagram_type, "Please select a diagram type.")

    response = chat(
        model='llava',
        messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}]
    )

    return response['message']['content']

# Main function to handle both student and reference image
def process_both_images(student_img, reference_img, diagram_type):
    student_answer = analyze_single_image(student_img, diagram_type)
    reference_answer = analyze_single_image(reference_img, diagram_type)

    # Save both answers to disk for further use
    with open("outputs/student_answer.txt", "w", encoding="utf-8") as f:
        f.write(student_answer)
    with open("outputs/reference_answer.txt", "w", encoding="utf-8") as f:
        f.write(reference_answer)

    # Combine and return as display
    combined = f"🧑‍🎓 **Student Answer:**\n{student_answer}\n\n📘 **Reference Answer:**\n{reference_answer}"
    return combined

# Gradio UI
demo = gr.Interface(
    fn=process_both_images,
    inputs=[
        gr.Image(type="filepath", label="Upload Student Image"),
        gr.Image(type="filepath", label="Upload Reference Image"),
        gr.Radio(
            choices=[
                "Flowchart", "Block Diagram", "Graph/Chart", "Table",
                "ER Diagram", "Network Diagram", "UML Diagram",
                "Circuit Diagram", "Scientific/Biological Diagram", "Mind Map/Concept Map"
            ],
            value="Flowchart",
            label="Select Diagram Type"
        )
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="Compare Student and Reference Diagrams",
    description="Upload both student and reference diagram images. Select the diagram type for accurate analysis and comparison."
)

if __name__ == "__main__":
    demo.launch()
