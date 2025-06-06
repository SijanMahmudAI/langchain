from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4o", temperature=0.7, max_completion_tokens=1024)

# Get prompt from user input
prompt = input("Please enter your prompt: ") # This will be the input to the chat model

# Generate a response using the chat model
result = model.invoke(prompt)

# Print the content of the result
print(result.content)