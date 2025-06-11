import gradio as gr
from ollama import chat

# Function to process image and prompt
def analyze_image_with_prompt(image, diagram_type):
    if image is None:
        return "Please provide an image."

    # If image is a file path, open and read as bytes
    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            img_bytes = img_file.read()
    else:
        img_bytes = image.read()

    # Adjust the prompt based on the selected diagram type
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
            f"the chart’s organization to enable accurate reproduction and evaluation."
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
            f"Accurately represent each component’s position, label, and connections to maintain the diagram’s semantic structure."
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
        messages=[
            {
                'role': 'user',
                'content': adjusted_prompt,
                'images': [img_bytes],
            }
        ]
    )

    return response['message']['content']

# Gradio interface
demo = gr.Interface(
    fn=analyze_image_with_prompt,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Radio(
            choices=[
                "Flowchart",
                "Block Diagram",
                "Graph/Chart",
                "Table",
                "ER Diagram",
                "Network Diagram",
                "UML Diagram",
                "Circuit Diagram",
                "Scientific/Biological Diagram",
                "Mind Map/Concept Map"
            ],
            value="Flowchart",
            label="Select Diagram Type"
        )
    ],
    outputs=gr.Textbox(label="Model Output"),
    title="Ollama Image Model Analyzer",
    description=(
        "Upload an image and select the type of diagram for analysis. "
        "The system will generate a tailored prompt based on your selection."
    )
)

if __name__ == "__main__":
    demo.launch()