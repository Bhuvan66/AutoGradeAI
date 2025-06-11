import gradio as gr
from ollama import chat

# Function to process image and prompt

def analyze_image_with_prompt(prompt, image):
    if image is None or prompt.strip() == "":
        return "Please provide both an image and a prompt."
    # If image is a file path, open and read as bytes
    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            img_bytes = img_file.read()
    else:
        img_bytes = image.read()
    response = chat(
        model='llava',
        messages=[
            {
                'role': 'user',
                'content': prompt,
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
        gr.Image(type="filepath", label="Upload Image")
    ],
    outputs=gr.Textbox(label="Model Output"),
    title="Ollama Image Model Analyzer",
    description="Upload an image and enter a prompt to analyze it using the Ollama model."
)

if __name__ == "__main__":
    demo.launch()
