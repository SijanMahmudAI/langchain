from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create llm model
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation")

# Initialize the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# Get prompt from the user input
prompt = input("Enter your prompt: ")

# Generate a response using the chat model
response = model.invoke(prompt)

# Print the response
print("Response from Hugging Face model:", response.content)