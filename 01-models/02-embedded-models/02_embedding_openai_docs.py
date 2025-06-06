from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings( model="text-embedding-3-small", dimensions=32)

# Example documentation for OpenAIEmbeddings:
documents = [
    "This is a document about AI.",
    "This document discusses machine learning.",
    "Here we talk about natural language processing."
]

# Example usage of the embeddings
result = embeddings.embed_documents(documents) # Output: [0.123, -0.456, ...] (example output, actual values will vary)

print(str(result))
