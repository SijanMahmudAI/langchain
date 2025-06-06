from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI LLM
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# Get prompt from user input
prompt = input("Please enter your prompt: ")

# Generate a response using the LLM
result = llm.invoke(prompt)

# Print the content of the result
print(result)