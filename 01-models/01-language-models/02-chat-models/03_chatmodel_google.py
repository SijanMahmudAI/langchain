from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Google chat model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Get prompt from the user input
prompt = input("Enter your prompt: ")

# Generate a response using the chat model
response = model.invoke(prompt)

# Print the response
print("Response from Google model:", response.content)
