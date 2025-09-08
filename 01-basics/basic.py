# Generated on Septemeper 08.
# Basic file to test the basic functionality of LangChain with Azure OpenAI.

# Allows the user to input his password or credentials without showing it on the console
import getpass
import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_VERSION= os.getenv("AZURE_OPEN_AI_API_VERSION")
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_DEPLOYMENT_VERSION
)

result=model.invoke("Hello World!")
print(result.content)
