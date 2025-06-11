import gradio as gr
from ollama import chat

# Function to process image and prompt
def analyze_image_with_prompt(prompt, image, diagram_type):
    if image is None  == "":
        return "Please provide both an image and a prompt."

    # If image is a file path, open and read as bytes
    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            img_bytes = img_file.read()
    else:
        img_bytes = image.read()

    # Adjust the prompt based on the selected diagram type
    if diagram_type == "Flowchart":
        adjusted_prompt = f"Analyze this flowchart: {prompt}"
    elif diagram_type == "Pie Chart":
        adjusted_prompt = f"Analyze this pie chart: {prompt}"
    elif diagram_type == "Bar Graph":
        adjusted_prompt = f"Analyze this bar graph: {prompt}"
    else:
        adjusted_prompt = prompt  # Default case, no change

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
        gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here..."),
        gr.Image(type="filepath", label="Upload Image"),
        gr.Radio(choices=["Flowchart", "Pie Chart", "Bar Graph"], value="Flowchart", label="Select Diagram Type")
    ],
    outputs=gr.Textbox(label="Model Output"),
    title="Ollama Image Model Analyzer",
    description="Upload an image and enter a prompt to analyze it using the Ollama model. Select the type of diagram for better analysis."
)

if __name__ == "__main__":
    demo.launch()