import os
import gradio as gr

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

AZURE_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
AZURE_EMBEDDINGS_KEY = os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY")
AZURE_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

AZURE_DEPLOYMENT_VERSION= os.getenv("AZURE_OPEN_AI_API_VERSION")


def get_llm():
    return AzureChatOpenAI(
        api_key=AZURE_KEY,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_version=AZURE_DEPLOYMENT_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

def get_azure_embeddings():
    return AzureOpenAIEmbeddings(
        api_key=AZURE_EMBEDDINGS_KEY,
        azure_endpoint=AZURE_EMBEDDINGS_ENDPOINT,
        api_version=AZURE_DEPLOYMENT_VERSION,
        azure_deployment=AZURE_EMBEDDINGS_DEPLOYMENT_NAME
    )

def get_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_docment = loader.load()
    return loaded_docment

def get_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def get_vectorstore(chunks):
    embedding_model = get_azure_embeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# Retriever 
def retriever(file):
    splits = get_loader(file)
    chunks = get_splitter(splits)
    vectordb = get_vectorstore(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(file, query):
    retriever_obj = retriever(file)
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever_obj,
                                     return_source_documents=False)
    response=qa.invoke(query)
    return response["result"]

rag_app = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF FILE", file_count="single", file_types=['.pdf'], type="file"),
        gr.Textbox(label="Input Query", lines=2, placeholder="How to pass the AZ-305 exam?")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG QA",
    description="Answer your PDF any question you want ever with simply load the file and then ask it whatever you want"
)

rag_app.launch(server_name="127.0.0.1", server_port=7726)    
