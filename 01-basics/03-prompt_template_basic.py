import getpass
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_VERSION= os.getenv("AZURE_OPEN_AI_API_VERSION")

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

model = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_DEPLOYMENT_VERSION
    )

# Read message from customer 

# Check if the user has typed Ctrl+C to exit the program
# Create list to hold conversation history
conversation_history = [
    SystemMessage(content="You are a mechanics shop owner that is very friendly helpful but biased to Infiniti and Nissan cars only and sometimes rude against other car brands. you can fix any car problem for those cars")
]

try:
    while True:
        message=input("Customer Message: ")
        
        conversation_history.append(HumanMessage(content=message))

        result=model.invoke(conversation_history)
        conversation_history.append(AIMessage(content=result.content))
        print("Shop Owner: ", result.content)
except KeyboardInterrupt:
    print("Exiting...")
    exit()

