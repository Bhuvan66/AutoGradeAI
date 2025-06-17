import gradio as gr

with gr.Blocks(css="custom_style.css") as demo:
    # ...existing code...
    pass  # your Gradio UI code here

if __name__ == "__main__":
    demo.launch()