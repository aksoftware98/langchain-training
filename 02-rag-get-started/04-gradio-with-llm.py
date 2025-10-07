import gradio as gr 

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_VERSION= os.getenv("AZURE_OPEN_AI_API_VERSION")

# Create the LLM 
model = AzureChatOpenAI(
    api_key=AZURE_KEY,
    api_version=AZURE_DEPLOYMENT_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME
)

template = PromptTemplate.from_template("Tell me a joke about {topic}")

def get_prompt(topic):
    return template.format(topic=topic)

def tell_joke(topic):
    result = model.invoke(get_prompt(topic))
    return result.content

# Build the gradio interface
demo = gr.Interface(
    fn=tell_joke,
    inputs=gr.Text(),
    outputs=gr.Text()
)

demo.launch(server_name="127.0.0.1", server_port=7860)