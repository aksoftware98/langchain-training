import os
from dotenv import load_dotenv 

load_dotenv()

AZURE_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
AZURE_EMBEDDINGS_KEY = os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY")
AZURE_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

AZURE_DEPLOYMENT_VERSION= os.getenv("AZURE_OPEN_AI_API_VERSION")

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader 
from langchain.chains import RetrievalQA

# Open the file that we want to embed 
file_content = None
with open("./mobile-policy.txt", "r") as f:
    file_content = f.read()
    print("Text has been founded successfully!")

embeddings_model = AzureOpenAIEmbeddings(
    api_key=AZURE_EMBEDDINGS_KEY,
    azure_endpoint=AZURE_EMBEDDINGS_ENDPOINT,
    azure_deployment=AZURE_EMBEDDINGS_DEPLOYMENT_NAME,
    api_version=AZURE_DEPLOYMENT_VERSION
)


# Create the splitter 
print("Splitting the text into chunks and loading it...")
loader = TextLoader("./mobile-policy.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f"Text has been splitted into {len(docs)} chunks successfully!")

# Create the vector store (in memory for demo purposes)
vectorstore = InMemoryVectorStore.from_documents(
    docs, 
    embedding=embeddings_model
)

# Create the chat model
model = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_DEPLOYMENT_VERSION
    )

qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=False)
print("You can now ask questions about the mobile phone policy. Type 'exit' to quit.")
try:
    while True:
        message=input("Your question: ")
        if message.lower() == 'exit':
            print("Exiting...")
            break
        result=qa.invoke(message)
        print("Answer: ", result)
except KeyboardInterrupt:
    print("Exiting...")
    exit()

