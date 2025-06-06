from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Anthropic chat model
model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)

# Get prompt from the user input
promt = input("Enter your prompt: ")

# Generate a response using the chat model
response = model.invoke(promt)

# Print the response
print("Response from Anthropic model:", response.content)
