import gradio as gr 

def add_numbers(a, b):
    return a + b

# Define the interface 
demo = gr.Interface(
    fn=add_numbers,
    inputs=[gr.Number(label="Number 1"), gr.Number(label="Number 2")],
    outputs=[gr.Number(label="Result")]
)

demo.launch()